import numpy as np
import torch
from torch.utils.data import Dataset

from mech_config import RUL_CAP_H


def rul_to_target(rul):
    return np.log1p(np.clip(rul, 0, RUL_CAP_H)) / np.log1p(RUL_CAP_H)


def target_to_rul(y):
    return np.expm1(np.clip(y, 0, 1) * np.log1p(RUL_CAP_H))


class VibrationDataset(Dataset):
    def __init__(self, path, feat_mean=None, feat_std=None, augment=False):
        z = np.load(path)
        self.signals = z["signals"]
        self.feats = z["feats"].astype(np.float32)
        self.cls, self.deg, self.rul = z["cls"].astype(np.int64), z["deg"].astype(np.float32), z["rul"].astype(np.float32)
        self.feat_mean = self.feats.mean(0) if feat_mean is None else np.asarray(feat_mean, np.float32)
        self.feat_std = (self.feats.std(0) + 1e-6) if feat_std is None else np.asarray(feat_std, np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.cls)

    def __getitem__(self, i):
        x = self.signals[i].astype(np.float32)
        if self.augment:
            rng = np.random.default_rng()
            x = np.roll(x, rng.integers(0, len(x))) * rng.uniform(0.8, 1.2)
            x = x + rng.normal(0, 0.02, len(x)).astype(np.float32)
        x = (x - x.mean()) / (x.std() + 1e-7)
        f = (self.feats[i] - self.feat_mean) / self.feat_std
        return {"input_values": torch.from_numpy(x.astype(np.float32)),
                "phys": torch.from_numpy(f.astype(np.float32)),
                "cls": torch.tensor(self.cls[i]),
                "deg": torch.tensor(self.deg[i]),
                "rul": torch.tensor(float(rul_to_target(self.rul[i])), dtype=torch.float32)}
    