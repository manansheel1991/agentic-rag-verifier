"""Human-only quality labels. Counts never come from the RAG pipeline's auditor."""
import csv
import json
import statistics
from pathlib import Path

ARMS = ("basic", "verified")
LABELS = ("correctness", "factual_claims", "unsupported_claims", "completeness",
          "citation_supported_claims", "abstention", "review_notes")


def write_review(artifact, path):
    rows = []
    gold = {q["id"]: q for q in artifact["dataset"]["questions"]}
    for pair in artifact["questions"]:
        question = gold[pair["id"]]
        row = {"run_id": artifact["run_id"], "mode": artifact["mode"], "question_id": pair["id"],
               **{key: question[key] for key in ("category", "question", "answerability", "reference_answer")},
               "required_facts": json.dumps(question["required_facts"]),
               "gold_evidence": json.dumps(question["gold_evidence"]),
               "retrieved_evidence": json.dumps(pair["retrieved_evidence"])}
        for arm in ARMS:
            value = pair[arm]
            row.update({f"{arm}_{key}": value[key] if key in ("answer", "outcome") else json.dumps(value[key])
                        for key in ("answer", "outcome", "emitted_claims", "error")})
            row[f"{arm}_unanswered_parts"] = json.dumps((value["pipeline"] or {}).get("unanswered_parts", []))
            row[f"{arm}_disagreements"] = json.dumps((value["pipeline"] or {}).get("disagreements", []))
            row[f"{arm}_latency_ms"] = value["latency_ms"]
            row[f"{arm}_query_cost"] = json.dumps(value["attributed_query_cost"])
            row.update({f"{arm}_{label}": "" for label in LABELS})
        rows.append(row)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _integer(value):
    if value == "":
        return None
    if not value.isdecimal():
        raise ValueError("Review counts must be nonnegative integers or blank")
    return int(value)


def read_review(artifact, path):
    pairs = {q["id"]: q for q in artifact["questions"]}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    seen = set()
    for row in rows:
        identity = row["question_id"]
        if identity in seen or identity not in pairs or row["run_id"] != artifact["run_id"]:
            raise ValueError("Review rows must match this run with no duplicate or unknown IDs")
        seen.add(identity)
        for arm in ARMS:
            value = pairs[identity][arm]
            if row[f"{arm}_answer"] != (value["answer"] or ""):
                raise ValueError("Answer text changed; review labels must refer to the saved output")
            for key, choices in {"correctness": ("", "correct", "incorrect"),
                                 "completeness": ("", "complete", "partial", "none"),
                                 "abstention": ("", "yes", "no")}.items():
                if row[f"{arm}_{key}"] not in choices:
                    raise ValueError(f"Invalid {arm}_{key} label")
            counts = {key: _integer(row[f"{arm}_{key}"]) for key in
                      ("factual_claims", "unsupported_claims", "citation_supported_claims")}
            for key in ("unsupported_claims", "citation_supported_claims"):
                if counts[key] is not None and (counts["factual_claims"] is None or counts[key] > counts["factual_claims"]):
                    raise ValueError("Label factual_claims first; other counts cannot exceed it")
            if value["error"] and (row[f"{arm}_correctness"] == "correct" or row[f"{arm}_abstention"] == "yes"):
                raise ValueError("Operational errors are incorrect outputs, not deliberate abstentions")
    if seen != set(pairs):
        raise ValueError("Missing review rows; leave labels blank rather than delete questions")
    return {row["question_id"]: row for row in rows}


def _rate(numerator, denominator, eligible, reviewed):
    return {"value": numerator / denominator if denominator else None,
            "numerator": numerator if denominator else None, "denominator": denominator,
            "eligible_questions": eligible, "reviewed_questions": reviewed}


def summarize(artifact, labels):
    gold = {q["id"]: q for q in artifact["dataset"]["questions"]}
    summary = {"run_id": artifact["run_id"], "mode": artifact["mode"], "notice": artifact["notice"],
               "complete": artifact["complete"], "code_version": artifact["code_version"],
               "corpus": artifact["corpus"], "configuration": artifact["configuration"],
               "indexing": artifact["indexing"], "configurations": {}}
    physical = list(artifact["indexing"]["calls"])
    for pair in artifact["questions"]:
        physical.extend(pair["shared_retrieval"]["calls"])
        for arm in ARMS:
            physical.extend(pair[arm]["generation_cost"]["calls"])
    from src.evaluation import usage
    summary["physical_run_cost"] = usage(physical)
    summary["query_cost_excludes_indexing"] = True
    summary["shared_retrieval_cost_is_attributed_to_each_arm_but_paid_once"] = True
    for arm in ARMS:
        outputs = [pair[arm] for pair in artifact["questions"]]
        reviewed = [labels[pair["id"]] for pair in artifact["questions"]]
        correctness = [row[f"{arm}_correctness"] for row in reviewed if row[f"{arm}_correctness"]]
        completeness = [row[f"{arm}_completeness"] for row in reviewed if row[f"{arm}_completeness"]]
        claims, unsupported, citation_claims, supported_citations = 0, 0, 0, 0
        unsupported_reviewed = citation_reviewed = 0
        for row in reviewed:
            total = _integer(row[f"{arm}_factual_claims"])
            bad = _integer(row[f"{arm}_unsupported_claims"])
            cited = _integer(row[f"{arm}_citation_supported_claims"])
            if total is not None and bad is not None:
                claims += total
                unsupported += bad
                unsupported_reviewed += 1
            if total is not None and cited is not None:
                citation_claims += total
                supported_citations += cited
                citation_reviewed += 1
        quality = {
            "answer_correctness": _rate(correctness.count("correct"), len(correctness), len(outputs), len(correctness)),
            "unsupported_claim_rate": _rate(unsupported, claims, len(outputs), unsupported_reviewed),
            "completeness": _rate(completeness.count("complete"), len(completeness), len(outputs), len(completeness)),
            "citation_support": _rate(supported_citations, citation_claims, len(outputs), citation_reviewed),
        }
        for name, answerability in (("abstention_on_unanswerable", "unanswerable"),
                                    ("false_refusals_on_answerable", "answerable")):
            subset = [row for row in reviewed if gold[row["question_id"]]["answerability"] == answerability]
            values = [row[f"{arm}_abstention"] for row in subset if row[f"{arm}_abstention"]]
            quality[name] = _rate(values.count("yes"), len(values), len(subset), len(values))
        latencies = [value["latency_ms"] for value in outputs]
        events = [event for pair in artifact["questions"] for event in
                  pair["shared_retrieval"]["calls"] + pair[arm]["generation_cost"]["calls"]]
        summary["configurations"][arm] = {"quality": quality, "queries": len(outputs),
            "errors": sum(bool(value["error"]) for value in outputs),
            "empty_outputs": sum(not value["answer"] for value in outputs),
            "partial_outputs": sum(value["outcome"] == "partial" for value in outputs),
            "mean_query_latency_ms": statistics.mean(latencies) if latencies else None,
            "median_query_latency_ms": statistics.median(latencies) if latencies else None,
            "attributed_query_cost": usage(events)}
    summary["paired_correctness_reviews"] = sum(all(labels[pair["id"]][f"{arm}_correctness"] for arm in ARMS)
                                               for pair in artifact["questions"])
    return summary


def report(results_path, review_path):
    results_path = Path(results_path)
    artifact = json.loads(results_path.read_text())
    summary = summarize(artifact, read_review(artifact, review_path))
    directory = results_path.parent
    (directory / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [f"# {summary['notice']}", "", "Quality scores use human labels only; unreviewed values remain unscored.",
             "Rates below show numerator/denominator; review coverage may differ between arms.", "",
             "| Configuration | Correctness | Unsupported claims | Abstention (unanswerable) | False refusals (answerable) | Mean query ms | Calls* | Available tokens* |",
             "|---|---|---|---|---|---|---|---|"]
    def rate(value):
        return "unscored" if value["value"] is None else f"{value['value']:.1%} ({value['numerator']}/{value['denominator']})"
    for name in ARMS:
        arm = summary["configurations"][name]
        q, costs = arm["quality"], arm["attributed_query_cost"]
        lines.append(f"| {name} | {rate(q['answer_correctness'])} | {rate(q['unsupported_claim_rate'])} | "
                     f"{rate(q['abstention_on_unanswerable'])} | {rate(q['false_refusals_on_answerable'])} | "
                     f"{arm['mean_query_latency_ms']:.2f} | {costs['model_calls']} | {costs['available_total_tokens']} |")
    lines += ["", "*Indexing excluded. Shared retrieval is attributed to both arms, physically executed once.",
              "Available token totals are partial when calls lack usage; None means unavailable, not zero.",
              "In MOCK mode, all outputs, timings and call counts describe scripted local fixtures, never model performance.",
              "See comparison.json for indexing costs, physical run costs, errors, empty/partial outputs, and review coverage."]
    (directory / "comparison.md").write_text("\n".join(lines) + "\n")
    return summary
