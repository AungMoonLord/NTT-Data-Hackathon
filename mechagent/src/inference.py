"""Step 3 – inference. Local class + SageMaker handlers (model_fn / input_fn / predict_fn / output_fn)."""
import json
import os
import sys
from math import gcd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import resample_poly
from transformers import AutoConfig

from bearing_physics import physics_check, physics_features
from mech_config import FS, N_SAMPLES
from mtl_model import MechAgentMTL


class MechAgentPredictor:
    def __init__(self, model_dir, device=None):
        model_dir = Path(model_dir)
        self.cfg = json.loads((model_dir / "mechagent_config.json").read_text())
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        bcfg = AutoConfig.from_pretrained(str(model_dir / "backbone_config"))
        self.model = MechAgentMTL(n_phys=self.cfg["n_phys"], n_classes=len(self.cfg["classes"]), backbone_config=bcfg)
        self.model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
        self.model.to(self.device).eval()
        self.mean = np.asarray(self.cfg["feat_mean"], np.float32)
        self.std = np.asarray(self.cfg["feat_std"], np.float32)
        self.cap = float(self.cfg["rul_cap_h"])

    def _to_rul(self, y):
        return round(float(np.expm1(np.clip(y, 0, 1) * np.log1p(self.cap))), 1)

    def _set_mc_dropout(self, on):
        self.model.eval()
        if on:   # Monte-Carlo dropout only in the heads → epistemic uncertainty
            for m in list(self.model.phys_branch.modules()) + list(self.model.shared.modules()):
                if isinstance(m, nn.Dropout):
                    m.train()

    @torch.no_grad()
    def predict(self, signal, rpm, fs=FS, bearing="6205", mc_samples=10, max_windows=4):
        x = np.asarray(signal, dtype=np.float64).ravel()
        if fs != FS:
            g = gcd(int(fs), FS)
            x = resample_poly(x, FS // g, int(fs) // g)
        if len(x) < N_SAMPLES // 4:
            raise ValueError("signal too short – need >= 0.25 s")
        if len(x) < N_SAMPLES:
            x = np.pad(x, (0, N_SAMPLES - len(x)), mode="wrap")
        wins = [x[i * N_SAMPLES:(i + 1) * N_SAMPLES] for i in range(min(max_windows, len(x) // N_SAMPLES))]

        feats = (np.stack([physics_features(w, FS, rpm, bearing) for w in wins]) - self.mean) / self.std
        xs = np.stack([(w - w.mean()) / (w.std() + 1e-7) for w in wins]).astype(np.float32)
        xt = torch.from_numpy(xs).to(self.device)
        ft = torch.from_numpy(feats.astype(np.float32)).to(self.device)

        self._set_mc_dropout(False)
        pooled = self.model.embed(xt)                  # heavy encoder: once
        self._set_mc_dropout(mc_samples > 1)
        outs = [self.model.heads(pooled, ft) for _ in range(max(1, mc_samples))]   # light heads: N times
        self._set_mc_dropout(False)

        probs = torch.stack([torch.softmax(o["logits"], -1) for o in outs]).mean((0, 1)).cpu().numpy()
        deg = torch.stack([o["deg"] for o in outs]).cpu().numpy()
        mu = torch.stack([o["rul_mu"] for o in outs]).cpu().numpy()
        var_ale = torch.stack([torch.exp(o["rul_logvar"]) for o in outs]).cpu().numpy()
        mu_m, sigma = float(mu.mean()), float(np.sqrt(var_ale.mean() + mu.var()))
        idx = int(probs.argmax())
        fault = self.cfg["classes"][idx]
        phys = physics_check(wins[0], FS, rpm, bearing)
        return {
            "fault": fault, "confidence": round(float(probs[idx]), 4),
            "class_probs": {c: round(float(p), 4) for c, p in zip(self.cfg["classes"], probs)},
            "degradation_pct": round(float(deg.mean()) * 100, 2),
            "degradation_std_pct": round(float(deg.std()) * 100, 2),
            "rul_hours": {"p05": self._to_rul(mu_m - 1.645 * sigma), "p50": self._to_rul(mu_m),
                          "p95": self._to_rul(mu_m + 1.645 * sigma), "cap": self.cap},
            "physics": phys, "ai_physics_agree": phys["physics_fault"] == fault,
            "rpm": float(rpm), "bearing": str(bearing), "n_windows": len(wins),
            "model": self.cfg["backbone_name"] + " + MTL heads",
        }


# ---------------- SageMaker Hugging Face DLC handlers ----------------
def model_fn(model_dir, context=None):
    return MechAgentPredictor(model_dir)


def input_fn(input_data, content_type="application/json", context=None):
    if isinstance(input_data, (bytes, bytearray)):
        input_data = input_data.decode("utf-8")
    return json.loads(input_data)


def predict_fn(data, model, context=None):
    return model.predict(signal=data["signal"], rpm=float(data["rpm"]), fs=int(data.get("fs", FS)),
                         bearing=str(data.get("bearing", "6205")), mc_samples=int(data.get("mc_samples", 10)))


def output_fn(prediction, accept="application/json", context=None):
    return json.dumps(prediction)


if __name__ == "__main__":
    import argparse
    from simulate import simulate_signal
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="artifacts/model")
    p.add_argument("--fault", default="inner_race")
    p.add_argument("--severity", type=float, default=0.6)
    p.add_argument("--rpm", type=float, default=1800)
    p.add_argument("--bearing", default="6205")
    a = p.parse_args()
    sig = simulate_signal(a.fault, a.severity, a.rpm, np.random.default_rng(), bearing=a.bearing)
    out = MechAgentPredictor(a.model_dir).predict(sig, a.rpm, bearing=a.bearing)
    out["physics"].pop("envelope_order_spectrum")
    print(json.dumps(out, indent=2, ensure_ascii=False))
