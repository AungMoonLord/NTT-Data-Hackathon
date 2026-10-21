"""Step 2 – fine-tune the HF backbone with multitask heads. Works locally and as a SageMaker training job."""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from bearing_physics import PHYS_FEATURE_NAMES
from dataset import VibrationDataset, target_to_rul
from mech_config import CLASSES, DEFAULT_BACKBONE, FS, N_SAMPLES, RUL_CAP_H
from mtl_model import MechAgentMTL, multitask_loss

CODE_FILES = ["inference.py", "mtl_model.py", "bearing_physics.py", "mech_config.py"]


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=os.environ.get("SM_CHANNEL_TRAIN", "data"))
    p.add_argument("--model_dir", default=os.environ.get("SM_MODEL_DIR", "artifacts/model"))
    p.add_argument("--backbone", default=DEFAULT_BACKBONE)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr_backbone", type=float, default=3e-5)
    p.add_argument("--lr_heads", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=2 if sys.platform.startswith("linux") else 0)
    p.add_argument("--max_train_batches", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def macro_f1(y, p, k):
    f1s = []
    for c in range(k):
        tp, fp, fn = ((p == c) & (y == c)).sum(), ((p == c) & (y != c)).sum(), ((p != c) & (y == c)).sum()
        if tp + fp + fn:
            f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else 0.0


@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    model.eval()
    P, Y, Dp, Dt, Rp, Rt = [], [], [], [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out = model(batch["input_values"], batch["phys"])
        P.append(out["logits"].float().argmax(-1).cpu().numpy()); Y.append(batch["cls"].cpu().numpy())
        Dp.append(out["deg"].float().cpu().numpy()); Dt.append(batch["deg"].cpu().numpy())
        Rp.append(out["rul_mu"].float().cpu().numpy()); Rt.append(batch["rul"].cpu().numpy())
    p, y = np.concatenate(P), np.concatenate(Y)
    dp, dt = np.concatenate(Dp), np.concatenate(Dt)
    rp, rt = target_to_rul(np.concatenate(Rp)), target_to_rul(np.concatenate(Rt))
    fz = rt < RUL_CAP_H * 0.99                      # samples inside the prediction horizon
    cm = np.zeros((len(CLASSES), len(CLASSES)), int)
    np.add.at(cm, (y, p), 1)
    return {"accuracy": float((p == y).mean()), "macro_f1": macro_f1(y, p, len(CLASSES)),
            "deg_mae_pct": float(np.abs(dp - dt).mean() * 100),
            "rul_mae_h": float(np.abs(rp - rt).mean()),
            "rul_mae_h_fault_zone": float(np.abs(rp - rt)[fz].mean()) if fz.any() else 0.0,
            "rul_rel_err_fault_zone": float((np.abs(rp - rt) / np.maximum(rt, 1))[fz].mean()) if fz.any() else 0.0,
            "confusion_matrix": cm.tolist()}


def save_artifacts(model, args, train_ds, model_dir):
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "model.pt")
    model.backbone.config.save_pretrained(str(model_dir / "backbone_config"))
    cfg = {"backbone_name": args.backbone, "classes": CLASSES, "n_phys": int(train_ds.feats.shape[1]),
           "phys_feature_names": PHYS_FEATURE_NAMES, "feat_mean": train_ds.feat_mean.tolist(),
           "feat_std": train_ds.feat_std.tolist(), "fs": FS, "n_samples": N_SAMPLES, "rul_cap_h": RUL_CAP_H}
    (model_dir / "mechagent_config.json").write_text(json.dumps(cfg, indent=2))
    code_dir = model_dir / "code"                     # SageMaker inference code inside model.tar.gz
    code_dir.mkdir(exist_ok=True)
    for f in CODE_FILES:
        shutil.copy(Path(__file__).resolve().parent / f, code_dir / f)
    (code_dir / "requirements.txt").write_text("scipy>=1.10\n")


def train(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    data_dir, model_dir = Path(args.data_dir), Path(args.model_dir)

    train_ds = VibrationDataset(data_dir / "train.npz", augment=True)
    val_ds = VibrationDataset(data_dir / "val.npz", train_ds.feat_mean, train_ds.feat_std)
    mk = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh, drop_last=sh,
                                   num_workers=args.num_workers, pin_memory=use_amp)
    train_dl, val_dl = mk(train_ds, True), mk(val_ds, False)

    model = MechAgentMTL(n_phys=train_ds.feats.shape[1], n_classes=len(CLASSES), backbone_name=args.backbone).to(device)
    bb = [p for p in model.backbone.parameters() if p.requires_grad]
    heads = [p for n, p in model.named_parameters() if not n.startswith("backbone.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": bb, "lr": args.lr_backbone}, {"params": heads, "lr": args.lr_heads}],
                            weight_decay=args.weight_decay)
    spe = min(len(train_dl), args.max_train_batches) if args.max_train_batches else len(train_dl)
    total = max(1, spe * args.epochs)
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup_ratio * total), total)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best = -1e9
    for epoch in range(args.epochs):
        model.train()
        run = []
        for i, batch in enumerate(train_dl):
            if args.max_train_batches and i >= args.max_train_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(batch["input_values"], batch["phys"])
            loss, parts = multitask_loss({k: v.float() for k, v in out.items()}, batch, model.task_log_vars)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            run.append(loss.item())
        v = evaluate(model, val_dl, device, use_amp)
        print(f"epoch={epoch + 1} train_loss={np.mean(run):.4f}; val_acc={v['accuracy']:.4f}; "
              f"val_f1={v['macro_f1']:.4f}; val_deg_mae={v['deg_mae_pct']:.2f}; "
              f"val_rul_mae_h={v['rul_mae_h_fault_zone']:.1f};", flush=True)
        score = v["macro_f1"] - v["deg_mae_pct"] / 100 - v["rul_mae_h_fault_zone"] / RUL_CAP_H
        if score > best:
            best = score
            save_artifacts(model, args, train_ds, model_dir)

    model.load_state_dict(torch.load(model_dir / "model.pt", map_location=device))
    results = {"val": evaluate(model, val_dl, device, use_amp)}
    for split in ("test", "test_zeroshot"):
        if (data_dir / f"{split}.npz").exists():
            ds = VibrationDataset(data_dir / f"{split}.npz", train_ds.feat_mean, train_ds.feat_std)
            results[split] = evaluate(model, mk(ds, False), device, use_amp)
    (model_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: {m: x for m, x in r.items() if m != "confusion_matrix"} for k, r in results.items()}, indent=2))
    return results


if __name__ == "__main__":
    train(parse_args())
