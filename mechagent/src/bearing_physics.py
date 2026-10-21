"""Bearing kinematics + envelope analysis (order domain) + physics verification."""
from dataclasses import dataclass

import numpy as np
from scipy import signal as sps
from scipy import stats

ENV_BAND_HZ = (1000.0, 5000.0)   # band-pass to isolate bearing impacts from machine noise
N_HARMONICS = 3
FAULT_KEYS = ["inner_race", "outer_race", "ball", "cage"]


@dataclass(frozen=True)
class BearingGeometry:
    name: str
    n_balls: int
    ball_d_mm: float
    pitch_d_mm: float
    contact_deg: float = 0.0

    def fault_orders(self) -> dict:
        """Characteristic fault frequencies as multiples of shaft speed (orders)."""
        r = self.ball_d_mm / self.pitch_d_mm * np.cos(np.radians(self.contact_deg))
        return {
            "inner_race": 0.5 * self.n_balls * (1 + r),                 # BPFI
            "outer_race": 0.5 * self.n_balls * (1 - r),                 # BPFO
            "ball": self.pitch_d_mm / (2 * self.ball_d_mm) * (1 - r ** 2),  # BSF
            "cage": 0.5 * (1 - r),                                      # FTF
        }


BEARINGS = {
    "6205": BearingGeometry("SKF 6205-2RS", 9, 7.94, 39.04),  # BPFI 5.42, BPFO 3.58, BSF 2.36, FTF 0.40
    "6208": BearingGeometry("SKF 6208", 9, 11.91, 60.0),       # approx. geometry (zero-shot test)
    "6310": BearingGeometry("SKF 6310", 8, 19.05, 80.0),       # approx. geometry (zero-shot test)
}


def get_bearing(key) -> BearingGeometry:
    return BEARINGS.get(str(key), BEARINGS["6205"])


def envelope_spectrum(x, fs, band=ENV_BAND_HZ):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    sos = sps.butter(4, [band[0], min(band[1], 0.45 * fs)], btype="bandpass", fs=fs, output="sos")
    env = np.abs(sps.hilbert(sps.sosfiltfilt(sos, x)))
    env -= env.mean()
    win = np.hanning(len(env))
    spec = np.abs(np.fft.rfft(env * win)) * 2.0 / win.sum()
    return np.fft.rfftfreq(len(env), 1.0 / fs), spec


def _peak_ratio(freqs, spec, f, tol=0.03):
    """Peak around f (±slip tolerance) divided by local median noise floor."""
    df = freqs[1] - freqs[0]
    if f <= 2 * df or f >= freqs[-1]:
        return 1.0
    half = max(tol * f, 1.5 * df)
    pk = spec[(freqs >= f - half) & (freqs <= f + half)]
    nh = max(0.2 * f, 15 * df)
    nb = spec[(freqs >= max(f - nh, df)) & (freqs <= f + nh)]
    return float(pk.max() / (np.median(nb) + 1e-12)) if pk.size else 1.0


def time_features(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    rms = float(np.sqrt(np.mean(x ** 2)) + 1e-12)
    peak = float(np.max(np.abs(x)))
    return {"rms": rms, "peak": peak, "crest": peak / rms,
            "kurtosis": float(stats.kurtosis(x, fisher=False)), "skewness": float(stats.skew(x))}


PHYS_FEATURE_NAMES = (["log_rms", "crest", "log_kurtosis", "skewness"]
                      + [f"{k}_h{h}" for k in FAULT_KEYS for h in range(1, N_HARMONICS + 1)]
                      + ["log_shaft_hz"])


def physics_features(x, fs, rpm, bearing="6205"):
    """17-dim speed-invariant feature vector fed into the model's physics branch."""
    tf = time_features(x)
    freqs, spec = envelope_spectrum(x, fs)
    fr = rpm / 60.0
    orders = get_bearing(bearing).fault_orders()
    feats = [np.log(tf["rms"]), tf["crest"], np.log(tf["kurtosis"] + 1e-6), tf["skewness"]]
    for k in FAULT_KEYS:
        for h in range(1, N_HARMONICS + 1):
            feats.append(np.log(_peak_ratio(freqs, spec, h * orders[k] * fr)))
    feats.append(np.log(fr))
    return np.asarray(feats, dtype=np.float32)


def physics_check(x, fs, rpm, bearing="6205", threshold=4.0, max_order=12.0, n_points=600):
    """Step 4: verify AI result against physical fault frequencies."""
    freqs, spec = envelope_spectrum(x, fs)
    fr = rpm / 60.0
    scores = {}
    for k, o in get_bearing(bearing).fault_orders().items():
        ratios = [_peak_ratio(freqs, spec, h * o * fr) for h in range(1, N_HARMONICS + 1)]
        detected = ratios[0] >= threshold and float(np.mean(ratios)) >= 0.6 * threshold
        scores[k] = {"order": round(float(o), 3), "freq_hz": round(float(o * fr), 2),
                     "harmonic_snr": [round(r, 2) for r in ratios], "detected": bool(detected)}
    det = [k for k, v in scores.items() if v["detected"]]
    physics_fault = max(det, key=lambda k: np.mean(scores[k]["harmonic_snr"])) if det else "normal"

    order_axis = freqs / fr
    m = order_axis <= max_order
    ox, sy = order_axis[m], spec[m]
    if len(ox) > n_points:  # max-pool so that peaks survive downsampling
        ox = np.array([c.mean() for c in np.array_split(ox, n_points)])
        sy = np.array([c.max() for c in np.array_split(sy, n_points)])
    return {"physics_fault": physics_fault, "threshold": threshold, "scores": scores,
            "time_features": time_features(x),
            "envelope_order_spectrum": {"order": [round(float(v), 3) for v in ox],
                                        "amp": [float(v) for v in sy]}}
