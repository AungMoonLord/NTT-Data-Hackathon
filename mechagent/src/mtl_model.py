"""Hard-parameter-sharing multitask model: pretrained HF audio encoder + physics branch → 3 heads."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class MechAgentMTL(nn.Module):
    def __init__(self, n_phys, n_classes=5, backbone_name=None, backbone_config=None,
                 freeze_feature_encoder=True, dropout=0.2):
        super().__init__()
        self.backbone = (AutoModel.from_config(backbone_config) if backbone_config is not None
                         else AutoModel.from_pretrained(backbone_name))
        fe = getattr(self.backbone, "feature_extractor", None)   # CNN front-end of HuBERT/wav2vec2
        if freeze_feature_encoder and fe is not None:
            for p in fe.parameters():
                p.requires_grad = False
        H = self.backbone.config.hidden_size
        self.att_pool = nn.Sequential(nn.Linear(H, 128), nn.Tanh(), nn.Linear(128, 1))
        self.phys_branch = nn.Sequential(nn.Linear(n_phys, 128), nn.GELU(), nn.Dropout(dropout),
                                         nn.Linear(128, 128), nn.GELU())
        self.shared = nn.Sequential(nn.LayerNorm(H + 128), nn.Linear(H + 128, 256), nn.GELU(), nn.Dropout(dropout))
        self.head_cls = nn.Linear(256, n_classes)   # fault location
        self.head_deg = nn.Linear(256, 1)           # degradation %
        self.head_rul = nn.Linear(256, 2)           # RUL mean + log-variance
        self.task_log_vars = nn.Parameter(torch.zeros(2))   # Kendall uncertainty weighting (cls, deg)

    def embed(self, input_values):
        hs = self.backbone(input_values=input_values).last_hidden_state   # (B, T, H)
        w = torch.softmax(self.att_pool(hs), dim=1)
        return (w * hs).sum(1)

    def heads(self, pooled, phys):
        z = self.shared(torch.cat([pooled, self.phys_branch(phys)], dim=-1))
        r = self.head_rul(z)
        return {"logits": self.head_cls(z),
                "deg": torch.sigmoid(self.head_deg(z)).squeeze(-1),
                "rul_mu": torch.sigmoid(r[:, 0]),
                "rul_logvar": r[:, 1].clamp(-8.0, 2.0)}

    def forward(self, input_values, phys):
        return self.heads(self.embed(input_values), phys)


def multitask_loss(out, batch, task_log_vars):
    l_cls = F.cross_entropy(out["logits"], batch["cls"], label_smoothing=0.05)
    l_deg = F.smooth_l1_loss(out["deg"], batch["deg"], beta=0.05)
    err2 = (out["rul_mu"] - batch["rul"]) ** 2
    l_rul = 0.5 * (torch.exp(-out["rul_logvar"]) * err2 + out["rul_logvar"]).mean() + err2.mean()
    s = task_log_vars.clamp(-3, 3)
    total = torch.exp(-s[0]) * l_cls + s[0] + torch.exp(-s[1]) * l_deg + s[1] + l_rul
    return total, {"cls": l_cls.item(), "deg": l_deg.item(), "rul": l_rul.item()}