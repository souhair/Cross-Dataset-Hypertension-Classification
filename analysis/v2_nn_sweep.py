# ===== Reproducibility seed — auto-injected by patch_analysis_seeds.py =====
import os as _os, random as _random, numpy as _np
_SEED = 42
_os.environ["PYTHONHASHSEED"] = str(_SEED)
_random.seed(_SEED)
_np.random.seed(_SEED)
try:
    import torch as _torch
    _torch.manual_seed(_SEED)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed_all(_SEED)
    _torch.backends.cudnn.deterministic = True
    _torch.backends.cudnn.benchmark = False
except ImportError:
    pass
# ==========================================================================
import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
for _sub in ("adapters", "extractors", "runners", "analysis",
             "figures", "audits", "diagnostics", "verification",
             "helpers"):
    _p = _os.path.join(_ROOT, _sub)
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
# ------------------------------------------------------------------



# v2_nn_sweep.py
"""
Neural network architecture comparison on Aurora-BP v2 flat features.

8 NN architectures + XGBoost reference.
5-fold StratifiedGroupKFold, subject-wise, with fold-aware preprocessing.
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score,
                              roc_auc_score, confusion_matrix,
                              classification_report)
from xgboost import XGBClassifier

from config import (SEED, N_FOLDS, XGB_3CLASS, NAN_THRESH, CORR_THRESH)
from v2_utils import load_aurora_raw, RESULTS, save_oof_predictions
from fold_preprocessor import FoldAwarePreprocessor
from v2_bootstrap import bootstrap_ci, format_ci

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 100
PATIENCE = 15

np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)

print(f"Device: {DEVICE}")


# ============================================================
# Dataset
# ============================================================
class TabDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ============================================================
# Architectures
# ============================================================
class SimpleMLP(nn.Module):
    def __init__(self, d_in, n=3, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, n))
    def forward(self, x): return self.net(x)


class DeepMLP(nn.Module):
    def __init__(self, d_in, n=3, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, n))
    def forward(self, x): return self.net(x)


class WideMLP(nn.Module):
    def __init__(self, d_in, n=3, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, n))
    def forward(self, x): return self.net(x)


class MDPIResidualBlock(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_out)
        self.bn1 = nn.BatchNorm1d(d_out)
        self.dr = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_out, d_out)
        self.bn2 = nn.BatchNorm1d(d_out)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
    def forward(self, x):
        s = self.skip(x)
        h = F.silu(self.bn1(self.fc1(x)))
        h = self.dr(h)
        h = self.bn2(self.fc2(h))
        return F.silu(h + s)


class ResidualMLP(nn.Module):
    def __init__(self, d_in, n=3, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_in, 128), nn.BatchNorm1d(128),
                                    nn.ReLU(), nn.Dropout(dropout))
        self.b1 = MDPIResidualBlock(128, 128, dropout)
        self.skip = nn.Linear(128, 64)
        self.b2 = MDPIResidualBlock(64, 64, dropout)
        self.head = nn.Linear(64, n)
    def forward(self, x):
        h = self.proj(x); h = self.b1(h); h = self.skip(h); h = self.b2(h)
        return self.head(h)


class ResidualMLPDeep(nn.Module):
    def __init__(self, d_in, n=3, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_in, 128), nn.BatchNorm1d(128),
                                    nn.ReLU(), nn.Dropout(dropout))
        self.b1 = MDPIResidualBlock(128, 128, dropout)
        self.b2 = MDPIResidualBlock(128, 128, dropout)
        self.b3 = MDPIResidualBlock(128, 64, dropout)
        self.head = nn.Linear(64, n)
    def forward(self, x):
        h = self.proj(x); h = self.b1(h); h = self.b2(h); h = self.b3(h)
        return self.head(h)


class SNN(nn.Module):
    def __init__(self, d_in, n=3, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.SELU(), nn.AlphaDropout(dropout),
            nn.Linear(256, 128), nn.SELU(), nn.AlphaDropout(dropout),
            nn.Linear(128, 64), nn.SELU(), nn.AlphaDropout(dropout),
            nn.Linear(64, n))
    def forward(self, x): return self.net(x)


class GatedBlock(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.3):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out)
        self.gate = nn.Linear(d_in, d_out)
        self.bn = nn.BatchNorm1d(d_out)
        self.dr = nn.Dropout(dropout)
    def forward(self, x):
        h = self.fc(x) * torch.sigmoid(self.gate(x))
        return self.dr(F.relu(self.bn(h)))


class GatedMLP(nn.Module):
    def __init__(self, d_in, n=3, dropout=0.3):
        super().__init__()
        self.b1 = GatedBlock(d_in, 128, dropout)
        self.b2 = GatedBlock(128, 64, dropout)
        self.head = nn.Linear(64, n)
    def forward(self, x):
        h = self.b1(x); h = self.b2(h)
        return self.head(h)


class TabNetLite(nn.Module):
    def __init__(self, d_in, d_model=64, n=3, n_steps=3, dropout=0.2):
        super().__init__()
        self.d_in = d_in
        self.n_steps = n_steps
        self.feat_embed = nn.Linear(1, d_model)
        self.step_fc = nn.Linear(d_in + d_model, d_model)
        self.attn = nn.Linear(d_model, d_in)
        self.head = nn.Linear(d_model, n)
        self.bn = nn.BatchNorm1d(d_model)
        self.dr = nn.Dropout(dropout)
    def forward(self, x):
        B = x.size(0)
        prior = torch.ones(B, self.d_in, device=x.device) / self.d_in
        out = torch.zeros(B, 64, device=x.device)
        h = torch.zeros(B, self.step_fc.in_features - self.d_in, device=x.device)
        for _ in range(self.n_steps):
            h_proj = F.relu(self.step_fc(torch.cat([x, h], dim=1)))
            attn_logits = self.attn(h_proj)
            mask = F.softmax(attn_logits + torch.log(prior + 1e-10), dim=-1)
            prior = prior * (2.0 - mask)
            prior = prior / prior.sum(dim=-1, keepdim=True).clamp(min=1e-10)
            feat = (x * mask).unsqueeze(-1)
            h = self.feat_embed(feat).sum(1)
            h = F.relu(self.bn(h))
            out = out + self.dr(h)
        return self.head(out)


ARCHS = {
    "SimpleMLP": lambda d: SimpleMLP(d),
    "DeepMLP": lambda d: DeepMLP(d),
    "WideMLP": lambda d: WideMLP(d),
    "ResidualMLP": lambda d: ResidualMLP(d),
    "ResidualMLPDeep": lambda d: ResidualMLPDeep(d),
    "SNN": lambda d: SNN(d),
    "GatedMLP": lambda d: GatedMLP(d),
    "TabNetLite": lambda d: TabNetLite(d),
}


# ============================================================
# Training
# ============================================================
def train_nn(model, X_tr, y_tr, X_va, y_va, lr=1e-3, wd=1e-4,
              batch_size=64, class_weights=None):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max",
                                                        factor=0.5, patience=6)
    if class_weights is not None:
        loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32).to(DEVICE))
    else:
        loss_fn = nn.CrossEntropyLoss()

    bs = min(batch_size, max(8, len(X_tr) // 2))
    dl = DataLoader(TabDS(X_tr, y_tr), batch_size=bs, shuffle=True,
                     drop_last=(len(X_tr) > bs))

    Xva = torch.tensor(X_va, dtype=torch.float32).to(DEVICE)
    yva = torch.tensor(y_va, dtype=torch.long).to(DEVICE)

    best_auc, best_state, wait = 0.0, None, 0
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(Xva)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        try:
            val_auc = roc_auc_score(y_va, probs, multi_class="ovr",
                                     average="macro")
        except Exception:
            val_auc = 0.0
        sched.step(val_auc)
        if val_auc > best_auc:
            best_auc, wait = val_auc, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model


def predict_nn(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32).to(DEVICE))
        return F.softmax(logits, dim=1).cpu().numpy()


# ============================================================
# CV with fold-aware preprocessing and grouped inner split
# ============================================================
def make_inner_split(y_tr, g_tr, val_frac=0.10, seed=SEED):
    """Subject-aware inner split for early stopping.

    Tries GroupShuffleSplit first. If that fails (e.g., a class is too small
    after the group shuffle), falls back to subject-level random selection
    — never a row-wise split.
    """
    n = len(y_tr)
    try:
        gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
        tr_rel, va_rel = next(gss.split(np.zeros(n), y_tr, groups=g_tr))
        # Require reasonable class coverage in both splits
        if (len(np.unique(y_tr[tr_rel])) >= 2 and
                len(np.unique(y_tr[va_rel])) >= 2):
            return tr_rel, va_rel
    except Exception:
        pass

    # Fallback: sample ~val_frac of unique subjects
    rng = np.random.default_rng(seed)
    unique_subj = np.unique(g_tr)
    n_val_subj = max(1, int(round(len(unique_subj) * val_frac)))
    val_subj = rng.choice(unique_subj, size=n_val_subj, replace=False)
    va_mask = np.isin(g_tr, val_subj)
    tr_rel = np.where(~va_mask)[0]
    va_rel = np.where(va_mask)[0]
    if len(tr_rel) == 0 or len(va_rel) == 0:
        # Ultra-safe fallback: split subjects deterministically by hash
        subj_sorted = np.sort(unique_subj)
        n_half = max(1, len(subj_sorted) // 2)
        val_subj = subj_sorted[-n_half:]
        va_mask = np.isin(g_tr, val_subj)
        tr_rel = np.where(~va_mask)[0]
        va_rel = np.where(va_mask)[0]
    return tr_rel, va_rel


def cv_evaluate(arch_name, arch_fn, df, y, g, CANDIDATE, n_folds=N_FOLDS):
    """5-fold StratifiedGroupKFold with subject-aware inner split and
    inner-train-only preprocessing."""
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True,
                                  random_state=SEED)
    fold_metrics = []
    all_y_true, all_y_pred, all_y_prob, all_subj = [], [], [], []

    for fold, (tr, te) in enumerate(sgkf.split(df[CANDIDATE].values, y,
                                                 groups=g), 1):
        # ----------------------------------------------------------------
        # 1. Inner split FIRST (subject-aware) — before any preprocessing
        # ----------------------------------------------------------------
        tr_rel, va_rel = make_inner_split(y[tr], g[tr])

        # ----------------------------------------------------------------
        # 2. Fit preprocessor on INNER-TRAIN only  (reviewer fix #7a)
        # ----------------------------------------------------------------
        pre_inner = FoldAwarePreprocessor(NAN_THRESH, CORR_THRESH).fit(
            df.iloc[tr].iloc[tr_rel], CANDIDATE)

        Xtr_raw = pre_inner.transform(df.iloc[tr].iloc[tr_rel]).astype(np.float32)
        Xva_raw = pre_inner.transform(df.iloc[tr].iloc[va_rel]).astype(np.float32)
        Xte_raw = pre_inner.transform(df.iloc[te]).astype(np.float32)

        yt = y[tr][tr_rel]
        yv = y[tr][va_rel]
        y_te = y[te]

        # ----------------------------------------------------------------
        # 3. Scaler fit on inner-train only
        # ----------------------------------------------------------------
        sc = StandardScaler().fit(Xtr_raw)
        Xtr = sc.transform(Xtr_raw)
        Xva = sc.transform(Xva_raw)
        Xte = sc.transform(Xte_raw)

        cw = len(yt) / (3 * np.maximum(np.bincount(yt, minlength=3), 1))

        model = arch_fn(Xtr.shape[1])
        model = train_nn(model, Xtr, yt, Xva, yv, class_weights=cw)
        prob = predict_nn(model, Xte)
        pred = prob.argmax(1)

        fold_metrics.append({
            "fold": fold,
            "auc": roc_auc_score(y_te, prob, multi_class="ovr", average="macro"),
            "macro_f1": f1_score(y_te, pred, average="macro", zero_division=0),
            "bal_acc": balanced_accuracy_score(y_te, pred),
            "acc": accuracy_score(y_te, pred),
        })
        all_y_true.extend(y_te.tolist())
        all_y_pred.extend(pred.tolist())
        all_y_prob.extend(prob.tolist())
        all_subj.extend(g[te].tolist())

    df_f = pd.DataFrame(fold_metrics)
    bootstrap_auc_mean = roc_auc_score(np.array(all_y_true), np.array(all_y_prob),
                                multi_class="ovr", average="macro")
    print(f"  {arch_name:18s}  AUC={df_f.auc.mean():.4f}±{df_f.auc.std():.4f}  "
          f"F1={df_f.macro_f1.mean():.4f}  "
          f"BAcc={df_f.bal_acc.mean():.4f}  "
          f"Acc={df_f.acc.mean():.4f}")

    return {
        "arch": arch_name,
        "auc_mean": df_f.auc.mean(), "auc_std": df_f.auc.std(),
        "f1_mean": df_f.macro_f1.mean(), "f1_std": df_f.macro_f1.std(),
        "bacc_mean": df_f.bal_acc.mean(), "bacc_std": df_f.bal_acc.std(),
        "acc_mean": df_f.acc.mean(), "acc_std": df_f.acc.std(),
        "bootstrap_auc_mean": bootstrap_auc_mean,
        "y_true": np.array(all_y_true),
        "y_pred": np.array(all_y_pred),
        "y_prob": np.array(all_y_prob),
        "y_subj": np.array(all_subj),
    }


def xgb_cv(df, y, g, CANDIDATE):
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                                  random_state=SEED)
    fold_metrics = []
    all_y_true, all_y_pred, all_y_prob, all_subj = [], [], [], []

    for fold, (tr, te) in enumerate(sgkf.split(df[CANDIDATE].values, y,
                                                 groups=g), 1):
        pre = FoldAwarePreprocessor(NAN_THRESH, CORR_THRESH).fit(
            df.iloc[tr], CANDIDATE)
        X_tr = pre.transform(df.iloc[tr])
        X_te = pre.transform(df.iloc[te])

        sc = StandardScaler().fit(X_tr)
        m = XGBClassifier(**XGB_3CLASS)
        m.fit(sc.transform(X_tr), y[tr])
        prob = m.predict_proba(sc.transform(X_te))
        pred = prob.argmax(1)

        fold_metrics.append({
            "fold": fold,
            "auc": roc_auc_score(y[te], prob, multi_class="ovr",
                                   average="macro"),
            "macro_f1": f1_score(y[te], pred, average="macro",
                                   zero_division=0),
            "bal_acc": balanced_accuracy_score(y[te], pred),
            "acc": accuracy_score(y[te], pred),
        })
        all_y_true.extend(y[te].tolist())
        all_y_pred.extend(pred.tolist())
        all_y_prob.extend(prob.tolist())
        all_subj.extend(g[te].tolist())

    df_f = pd.DataFrame(fold_metrics)
    return {
        "arch": "XGBoost",
        "auc_mean": df_f.auc.mean(), "auc_std": df_f.auc.std(),
        "f1_mean": df_f.macro_f1.mean(), "f1_std": df_f.macro_f1.std(),
        "bacc_mean": df_f.bal_acc.mean(), "bacc_std": df_f.bal_acc.std(),
        "acc_mean": df_f.acc.mean(), "acc_std": df_f.acc.std(),
        "bootstrap_auc_mean": roc_auc_score(all_y_true, all_y_prob,
                                     multi_class="ovr", average="macro"),
        "y_true": np.array(all_y_true),
        "y_pred": np.array(all_y_pred),
        "y_prob": np.array(all_y_prob),
        "y_subj": np.array(all_subj),
    }


# ============================================================
# Main
# ============================================================
def main():
    import pandas as pd  # local to avoid top-level import ordering

    print("Loading Aurora data...")
    df, CANDIDATE = load_aurora_raw()
    y = df.htn3.values
    g = df.subject_id.astype(str).values
    print(f"  shape: {len(df)} segments, {df.subject_id.nunique()} subjects, "
          f"{len(CANDIDATE)} candidate features")
    print(f"  classes: {np.bincount(y)}")

    results = []

    print("\n--- Reference: XGBoost ---")
    xgb_res = xgb_cv(df, y, g, CANDIDATE)
    print(f"  {'XGBoost':18s}  AUC={xgb_res['auc_mean']:.4f}±{xgb_res['auc_std']:.4f}  "
          f"F1={xgb_res['f1_mean']:.4f}  "
          f"BAcc={xgb_res['bacc_mean']:.4f}  "
          f"Acc={xgb_res['acc_mean']:.4f}")
    results.append(xgb_res)

    print("\n--- Neural network architectures ---")
    for name, fn in ARCHS.items():
        r = cv_evaluate(name, fn, df, y, g, CANDIDATE)
        results.append(r)

    # Summary
    rows = []
    for r in results:
        rows.append({
            "architecture": r["arch"],
            "AUC_mean": r["auc_mean"], "AUC_std": r["auc_std"],
            "MacroF1_mean": r["f1_mean"], "MacroF1_std": r["f1_std"],
            "BalAcc_mean": r["bacc_mean"], "BalAcc_std": r["bacc_std"],
            "Acc_mean": r["acc_mean"], "Acc_std": r["acc_std"],
            "pooled_AUC": r["bootstrap_auc_mean"],
        })
    df_summary = pd.DataFrame(rows).sort_values("AUC_mean", ascending=False)
    df_summary.to_csv(os.path.join(RESULTS, "v2_nn_sweep_summary.csv"),
                       index=False)

    print("\n" + "=" * 100)
    print("NEURAL NETWORK SWEEP — SUMMARY")
    print("=" * 100)
    print(df_summary.round(4).to_string(index=False))

    # Bootstrap for best NN and XGBoost
    print("\n" + "=" * 88)
    print("BOOTSTRAP 95% CI (1000 subject-stratified resamples)")
    print("=" * 88)
    nn_results = [r for r in results if r["arch"] != "XGBoost"]
    best_nn = max(nn_results, key=lambda r: r["auc_mean"])

    for r in [xgb_res, best_nn]:
        ci = bootstrap_ci(r["y_true"], r["y_pred"], r["y_prob"],
                           stratify_by=r["y_subj"], n_boot=1000, seed=SEED)
        print(f"\n  {r['arch']}")
        print(f"    AUC:      {format_ci(ci, 'auc')}")
        print(f"    Macro F1: {format_ci(ci, 'macro_f1')}")
        print(f"    BalAcc:   {format_ci(ci, 'bacc')}")
        print(f"    Acc:      {format_ci(ci, 'acc')}")

    # Save predictions
    np.savez_compressed(
        os.path.join(RESULTS, "v2_nn_sweep_predictions.npz"),
        **{r["arch"] + "_y_true": r["y_true"] for r in results},
        **{r["arch"] + "_y_pred": r["y_pred"] for r in results},
        **{r["arch"] + "_y_prob": r["y_prob"] for r in results},
        **{r["arch"] + "_y_subj": r["y_subj"] for r in results},
    )
    print("\nSaved:")
    print(f"  {os.path.join(RESULTS, 'v2_nn_sweep_summary.csv')}")
    print(f"  {os.path.join(RESULTS, 'v2_nn_sweep_predictions.npz')}")


if __name__ == "__main__":
    import pandas as pd
    main()