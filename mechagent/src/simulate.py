"""Step 1 – physics-based run-to-failure vibration simulator.
Models: periodic impacts at BPFI/BPFO/BSF/FTF, structural resonance, slip/jitter,
amplitude modulation (load zone), shaft harmonics, interference tones, random knocks,
background noise, and non-monotonic exponential degradation.
Usage: python src/simulate.py --out data --units 200 --snapshots 25
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

from bearing_physics import get_bearing, physics_features
from mech_config import CLASSES, FAULT_TYPES, FS, N_SAMPLES, RPM_RANGE, RUL_CAP_H

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **_):
        return x


def resonance_kernel(fs, fn, zeta, length_s=0.02):
    t = np.arange(int(length_s * fs)) / fs
    return np.exp(-2 * np.pi * zeta * fn * t) * np.sin(2 * np.pi * fn * t)


def fault_impulse_train(fault, f_fault, fr, ftf_hz, amp, rng, fs, n):
    period, slip = 1.0 / f_fault, rng.uniform(0.005, 0.02)
    times, tk = [], rng.uniform(0, period)
    while tk < n / fs:
        times.append(tk)
        tk += period * (1 + rng.normal(0, slip))          # slip / jitter
    times = np.asarray(times)
    a = np.full(times.shape, amp)
    phi, m = rng.uniform(0, 2 * np.pi), rng.uniform(0.4, 0.9)
    if fault == "inner_race":                             # passes through load zone once per rev
        a *= 1 + m * np.cos(2 * np.pi * fr * times + phi)
    elif fault == "ball":                                 # ball carried by the cage
        a *= 1 + m * np.cos(2 * np.pi * ftf_hz * times + phi)
    elif fault == "cage":
        a *= 1 + 0.5 * m * np.cos(2 * np.pi * fr * times + phi)
    a = np.clip(a, 0, None) * rng.uniform(0.7, 1.3, size=times.shape)
    imp = np.zeros(n)
    np.add.at(imp, np.clip((times * fs).astype(int), 0, n - 1), a)
    kern = resonance_kernel(fs, rng.uniform(2000, 4500), rng.uniform(0.02, 0.08))
    return fftconvolve(imp, kern)[:n]


def simulate_signal(fault, severity, rpm, rng, fs=FS, n=N_SAMPLES, bearing="6205"):
    """severity in [0,1] (0 = new, 1 = failure). Returns acceleration signal (g)."""
    t = np.arange(n) / fs
    fr = rpm / 60.0
    orders = get_bearing(bearing).fault_orders()
    x = rng.uniform(0.05, 0.3) * np.sin(2 * np.pi * fr * t + rng.uniform(0, 2 * np.pi))       # 1X
    x += rng.uniform(0.0, 0.1) * np.sin(2 * np.pi * 2 * fr * t + rng.uniform(0, 2 * np.pi))   # 2X
    for _ in range(rng.integers(0, 3)):                                                       # interference
        x += rng.uniform(0.01, 0.06) * np.sin(2 * np.pi * rng.uniform(50, 6000) * t + rng.uniform(0, 6.28))
    x += rng.normal(0, rng.uniform(0.05, 0.15) * (1 + 1.5 * severity), n)                     # noise grows with wear
    for _ in range(rng.poisson(1.0)):                                                         # random knocks
        k = rng.integers(0, n - 400)
        kern = resonance_kernel(fs, rng.uniform(1500, 6000), 0.05)
        x[k:k + len(kern)] += rng.uniform(0.1, 0.5) * kern
    if fault != "normal":
        amp = (0.08 + 0.92 * severity) * rng.uniform(0.6, 1.4)
        x += fault_impulse_train(fault, orders[fault] * fr, fr, orders["cage"] * fr, amp, rng, fs, n)
        if fault == "cage":
            x += 0.3 * amp * np.sin(2 * np.pi * orders["cage"] * fr * t)
    return x.astype(np.float32)


def degradation(tau, b, wobble_f, wobble_phase):
    d = (np.exp(b * tau) - 1) / (np.exp(b) - 1)
    d += 0.03 * tau * np.sin(2 * np.pi * wobble_f * tau + wobble_phase)   # non-monotonic wear
    return float(np.clip(d, 0, 1))


def generate_units(n_units, snapshots, seed, bearings=("6205",), healthy_frac=0.3, desc="units"):
    rng = np.random.default_rng(seed)
    rec = {k: [] for k in ["signals", "feats", "cls", "deg", "rul", "rpm", "unit", "time_h", "bearing"]}
    for u in tqdm(range(n_units), desc=desc):
        fault = FAULT_TYPES[u % len(FAULT_TYPES)]
        bearing = str(rng.choice(bearings))
        base_rpm = rng.uniform(*RPM_RANGE)
        t_on, t_prop = rng.uniform(200, 2000), rng.uniform(72, 720)   # hours
        life = t_on + t_prop
        b, wf, wp = rng.uniform(2, 5), rng.uniform(1, 3), rng.uniform(0, 2 * np.pi)
        n_h = int(round(snapshots * healthy_frac))
        times = np.concatenate([np.sort(rng.uniform(0, t_on, n_h)),
                                np.sort(rng.uniform(t_on, life * 0.999, snapshots - n_h))])
        for th in times:
            if th < t_on:
                cls, deg = "normal", 0.05 * th / t_on
            else:
                cls, deg = fault, 0.05 + 0.95 * degradation((th - t_on) / t_prop, b, wf, wp)
            rpm = base_rpm * (1 + rng.normal(0, 0.01))
            x = simulate_signal(cls, deg, rpm, rng, bearing=bearing)
            rec["signals"].append(x.astype(np.float16))
            rec["feats"].append(physics_features(x, FS, rpm, bearing))
            rec["cls"].append(CLASSES.index(cls))
            rec["deg"].append(deg)
            rec["rul"].append(min(life - th, RUL_CAP_H))
            rec["rpm"].append(rpm)
            rec["unit"].append(u)
            rec["time_h"].append(th)
            rec["bearing"].append(bearing)
    return {"signals": np.stack(rec["signals"]), "feats": np.stack(rec["feats"]),
            "cls": np.array(rec["cls"], np.int64), "deg": np.array(rec["deg"], np.float32),
            "rul": np.array(rec["rul"], np.float32), "rpm": np.array(rec["rpm"], np.float32),
            "unit": np.array(rec["unit"], np.int32), "time_h": np.array(rec["time_h"], np.float32),
            "bearing": np.array(rec["bearing"])}


def build_dataset(out_dir, n_units=200, snapshots=25, zeroshot_units=40, seed=42):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_tr, n_va = max(1, int(n_units * 0.7)), max(1, int(n_units * 0.15))
    splits = {"train": (n_tr, seed, ("6205",)), "val": (n_va, seed + 1, ("6205",)),
              "test": (max(1, n_units - n_tr - n_va), seed + 2, ("6205",))}
    if zeroshot_units:
        splits["test_zeroshot"] = (zeroshot_units, seed + 3, ("6208", "6310"))   # unseen bearings
    meta = {}
    for name, (n, s, b) in splits.items():   # split BY UNIT → no leakage between train/test
        d = generate_units(n, snapshots, s, b, desc=name)
        np.savez(out_dir / f"{name}.npz", **d)
        meta[name] = {"units": n, "samples": int(len(d["cls"])),
                      "class_counts": {c: int((d["cls"] == i).sum()) for i, c in enumerate(CLASSES)}}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data")
    p.add_argument("--units", type=int, default=200)
    p.add_argument("--snapshots", type=int, default=25)
    p.add_argument("--zeroshot_units", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    build_dataset(a.out, a.units, a.snapshots, a.zeroshot_units, a.seed)