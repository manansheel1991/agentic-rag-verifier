# Synthetic development evaluation — review required

This is **synthetic development data, not an independent benchmark**. The six short
corpus documents describe fictional Harbor and Cedar projects. The 25 questions,
reference answers, answerability labels, required facts and gold evidence need your
review before drawing conclusions. The corpus and dataset are redistributable under
CC0-1.0; see [LICENSE](LICENSE). No model performance results are bundled.

There are five questions in each category: direct facts, multiple passages, absent
answers, incorrect numerical premises, and conflicting evidence. Incorrect premises
are answerable by correcting them, not by inventing a rationale. Conflicted Cedar
questions require both attributed values and an explicit unresolved disagreement;
neither memo has precedence. Gold references identify documents and exact character
spans in source text, independently of any embedding model or chunk layout.

## Run

From the repository root, after installing `requirements-dev.txt`:

```bash
# Offline plumbing check. No API keys, model downloads, or paid calls.
python -m src.evaluation run --mode mock --output eval_runs/mock-dev
```

The mock provider produces scripted answers unrelated to the question's gold label,
and its auditor always withholds claims. **Mock outputs, call counts and local
wall-clock times are fixtures, not measured model performance.** Mock token usage is
unavailable rather than invented. Mock mode overrides model IDs with explicit mock
identities, blocks Python network connections, and still exercises real local Chroma.

Run live evaluation only when you deliberately want provider calls and accept their
cost. Set the needed key in your ignored `.env`, then use a different output directory:

```bash
# LIVE: performs API calls and can incur charges. This is not run by the tests.
python -m src.evaluation run --mode live --output eval_runs/live-dev \
  --embedding-model openai/text-embedding-3-small \
  --generation-model openai/gpt-4o-mini
```

Both commands accept `--chunk-size` (500), `--overlap` (100), `--top-k` (3) and
`--dataset` (defaults to this directory's `dataset.json`). Existing output directories
are refused so re-running cannot erase review labels. All generated runs and indexes
under `eval_runs/` are ignored by Git. A custom output directory should also be ignored
or outside the repository. Generation settings are fixed to the existing generator's
temperature 0 and structured schema; provider output-token defaults are unchanged.

## What is compared

Only two configurations are run, without answer regeneration or recovery:

- **A/basic:** initial retrieval → the existing structured claim generator → formatting.
  It does not run relevance verification, claim auditing, or citation-support gates.
- **B/verified:** the same initial retrieval → existing relevance verification → the
  same structured generator → existing claim audit and deterministic citation gates.

Both use all six documents, one embedding collection/configuration, identical ordered
initial evidence per question, the same generation model, prompt/schema and temperature,
and the same answer/citation renderer. The verifier may subsequently reject context
or claims; that is the behavior being compared. An empty citation list is representable
in the common claim format: A preserves it for review, while B's citation gate blocks
it. Formatting never infers that an unknown or missing citation makes a statement false.

Retrieval is executed **once per question** and its result is shared. A/B execution
order alternates by question. Generation fallback is disabled for this evaluation so
one arm cannot silently use a different generation model. Existing bounded transport
retries remain enabled and counted; they are not answer recovery. Normal CLI provider
fallback behavior is unchanged. Temperature 0 does not guarantee identical remote model
responses; this small single-run comparison does not establish statistical significance.

## Artifacts and cost accounting

- `results.json`: outputs side by side with emitted claims, drafts, verifier diagnostics,
  retrieved chunk text/IDs, errors, timings, actual provider-attempt model IDs and available
  usage. It embeds the dataset, corpus/file hashes, dataset hash, requested configuration,
  Git commit/dirty state and hashes of Python source and dependency declarations. This
  captures uncommitted code too. `complete: false` identifies an interrupted run.
- `review.csv`: one row per question with A and B outputs, references, retrieved evidence,
  and **blank** human review fields. Review the actual final presentation: answer plus
  surfaced source disagreements. Rejected drafts and internal audit reasoning are
  diagnostics, not emitted answer claims. Include factual assertions in any displayed
  disagreement explanations in your human counts.
- `comparison.json` and `comparison.md`: comparison table and detailed denominators,
  review coverage, error/empty/partial-output counts, timings and usage. No quality score
  is populated by reference matching or by the pipeline's auditor.

Indexing runs once into a fresh local index and is reported separately. Query latency
is shared retrieval wall time plus the arm's subsequent wall time, including retries
and local processing; indexing is excluded. Mean and median include failed queries.
This is an attribution for comparing standalone configurations, not a parallel speed test.

An actual SDK invocation counts as one model call, including failed/retried attempts.
Missing credentials cause zero SDK calls. `attributed_query_cost` assigns each arm the
shared retrieval calls plus its own generation/relevance/audit calls. Because retrieval
is physically shared, do not add both arms' attributed counts to calculate the bill.
`physical_run_cost` counts indexing once, shared retrieval once, and both arms' actual
calls. No dollar estimates are made. Available total tokens are summed only where the
provider supplies usage; `calls_with_usage` / `calls_without_usage` describe coverage.
`None` means unavailable or no reported usage, not zero. Failed calls may have unknown
usage, so known totals need not equal billable tokens.

## Human review and metrics

First review the synthetic references against the corpus. If you revise the dataset,
start a new run so its recorded hash matches. Review all 25 questions for **both** arms,
preferably with arm names concealed and order shuffled outside the saved files. Do not
copy verdicts from the system auditor into evaluation labels.

Edit only the label columns in `review.csv`; do not edit output text, run IDs or question
IDs. Allowed fields for each `basic_` / `verified_` prefix:

| Field | Values and instructions |
|---|---|
| `correctness` | `correct` or `incorrect`: assess the whole response against the reference and all required facts, including premise corrections or appropriate abstention/conflict reporting. |
| `factual_claims` | Nonnegative count of all factual propositions presented to the user, including **uncited** claims. Split compound claims if needed; do not trust emitted-list length. |
| `unsupported_claims` | Number of those propositions unsupported by or contradicted by the selected corpus. A missing citation alone does **not** make a true corpus-supported proposition unsupported or incorrect. |
| `completeness` | `complete`, `partial`, or `none`, according to required facts. |
| `citation_supported_claims` | Number of factual propositions whose attached citations actually support them. Uncited claims get no citation-support credit, independently of correctness. |
| `abstention` | `yes` for a deliberate full refusal/no substantive answer, otherwise `no`. A nonempty partial answer is not a full refusal. |
| `review_notes` | Optional explanation, especially for ambiguous references, conflicts or partial answers. |

Leave undecided fields blank. Counts of unsupported or citation-supported propositions
cannot exceed `factual_claims`. Operational errors are **incorrect**, `none` complete,
zero factual claims and **not** deliberate abstentions (`abstention=no`). A successful
empty output on an answerable question is an incorrect full abstention; on an absent
question an appropriate explicit refusal/empty answer can be correct and complete.
For conflicted questions, an empty scalar answer with correctly surfaced, referenced
source disagreements can be correct. Evaluate whether the full response supplies the
required conflict information; do not rely on its self-reported outcome.

Regenerate the comparison without any API calls:

```bash
python -m src.evaluation report \
  --results eval_runs/live-dev/results.json \
  --review eval_runs/live-dev/review.csv
```

The same report command works for mock runs, but those results remain labelled MOCK
and cannot be presented as model performance. The report rejects duplicate, foreign,
missing or stale output rows and invalid labels. Partial labeling is allowed; missing
labels never silently become passes or failures.

| Metric | Numerator / denominator |
|---|---|
| Answer correctness | Human `correct` responses / responses with a correctness label. |
| Unsupported-claim rate | Sum of human unsupported factual propositions / sum of human factual propositions, using rows with both counts. Uncited propositions are included. |
| Abstention on unanswerable | Human `abstention=yes` / abstention-labelled **unanswerable** questions (5 eligible). |
| False refusals on answerable | Human `abstention=yes` / abstention-labelled **answerable** questions (15 eligible, including incorrect premises). |
| Completeness | Human `complete` / completeness-labelled responses. |
| Citation support | Human citation-supported propositions / human factual propositions, using rows with both counts. |

Conflicted questions (5) participate in correctness and claim metrics but are excluded
from the two answerability-based refusal rates. Partial responses contribute every
emitted factual proposition to claim metrics; they are correct only if the actual
response satisfies the reference/required facts, and are not full refusals. Errors stay
in the question set: label them as above so they contribute failures to correctness and
`no` to refusal-rate numerators, not misleading successful abstentions. Operational
error counts are always shown separately, including when human labels are absent.

All human-dependent rates remain `null`/`unscored` until their required labels exist.
A zero factual-claim denominator also stays unscored, never 0% unsupportedness. Every
metric records its numerator, denominator, eligible-question count and reviewed-question
count. A partially reviewed table is a **reviewed-subset result**, not an estimate for
all 25 questions. Compare arms only after reviewing the same questions; the JSON also
reports paired correctness-review coverage. Empty/partial statuses and latency/calls
are operational observations, not substitutes for human correctness or completeness.

## Offline tests

```bash
python -m pytest -q
```

Focused tests check data/evidence integrity, identical generator inputs and formatting,
skipping verification in A, uncited-claim retention, mock isolation, attempt/usage
accounting, no automatic quality scores, human metric denominators, partial/error/empty
handling and stale review rejection. Mocks validate mechanics, **not** the accuracy of
model judgments. Retrieval misses, truncated passages, model errors and imperfect
synthetic references can all affect live results.
