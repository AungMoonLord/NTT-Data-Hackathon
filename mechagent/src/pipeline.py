"""Full local pipeline: 1) simulate → 2) train → 3) inference → 4) report.
Smoke test: python src/pipeline.py --quick     Full: python src/pipeline.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
import simulate  # noqa: E402
import train  # noqa: E402
from inference import MechAgentPredictor  # noqa: E402
from report import build_report  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--model_dir", default="artifacts/model")
    p.add_argument("--out", default="outputs")
    p.add_argument("--skip_sim", action="store_true")
    p.add_argument("--skip_train", action="store_true")
    a = p.parse_args()

    if not a.skip_sim:
        print("=== [1] Simulation ===")
        simulate.build_dataset(a.data_dir, n_units=40 if a.quick else 200,
                               snapshots=10 if a.quick else 25, zeroshot_units=0 if a.quick else 40)
    if not a.skip_train:
        print("=== [2] Training ===")
        targs = ["--data_dir", a.data_dir, "--model_dir", a.model_dir,
                 "--epochs", "1" if a.quick else "8", "--batch_size", "8" if a.quick else "16"]
        train.train(train.parse_args(targs + (["--max_train_batches", "30"] if a.quick else [])))

    print("=== [3] Inference + [4] Report ===")
    predictor = MechAgentPredictor(a.model_dir)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    for fault, sev, rpm in [("normal", 0.02, 1500), ("inner_race", 0.75, 1800),
                            ("outer_race", 0.45, 2400), ("ball", 0.3, 1200), ("cage", 0.9, 3000)]:
        x = simulate.simulate_signal(fault, sev, rpm, rng)
        pred = predictor.predict(x, rpm)
        rep = build_report(pred, {"machine_id": f"DEMO-{fault}", "name": "Demo machine",
                                  "bearing_model": "SKF 6205-2RS"}, job_id=f"local-{fault}")
        (out / f"report_{fault}.md").write_text(rep["markdown"], encoding="utf-8")
        s = rep["summary"]
        print(f"[truth={fault:<10} sev={sev:.2f}] pred={s['fault']:<10} conf={s['confidence']:.2f} "
              f"deg={s['degradation_pct']:.1f}% replace_within={s['replace_within']['text']}")
    print(f"Reports saved to {out}/")
