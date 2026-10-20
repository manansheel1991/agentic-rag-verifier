# Harbor operations (synthetic)

The Harbor service processes 420 events per second at peak. Its target latency is 80 milliseconds.

Harbor replicas run in us-east-1 and us-west-2. Disaster recovery drills occur every two weeks on Sunday at 02:00 UTC.

Harbor retains event logs for 14 days. The on-call engineer acknowledges critical alerts within 15 minutes.
