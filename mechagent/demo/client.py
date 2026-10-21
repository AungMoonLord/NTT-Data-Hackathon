"""CLI demo: simulate a signal → call API → print the report."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simulate import simulate_signal  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--api", required=True)
p.add_argument("--machine", default="PUMP-01")
p.add_argument("--fault", default="inner_race")
p.add_argument("--severity", type=float, default=0.7)
p.add_argument("--rpm", type=float, default=1800)
p.add_argument("--bearing", default="6205")
a = p.parse_args()

x = simulate_signal(a.fault, a.severity, a.rpm, np.random.default_rng(), bearing=a.bearing)
r = requests.post(f"{a.api.rstrip('/')}/analyze", timeout=30,
                  json={"machine_id": a.machine, "rpm": a.rpm, "fs": 16000, "bearing": a.bearing,
                        "signal": np.round(x, 4).tolist()})
r.raise_for_status()
job_id = r.json()["job_id"]
print("Job:", job_id)
for _ in range(60):
    job = requests.get(f"{a.api.rstrip('/')}/reports/{job_id}", timeout=15).json()["job"]
    print("status:", job["status"])
    if job.get("done"):
        break
    time.sleep(5)
print(job.get("report", {}).get("markdown", job.get("error")))
if job.get("agent"):
    print("\n--- Agent ---\n", job["agent"]["final_text"], "\nCost:", job["agent"]["usage"])
