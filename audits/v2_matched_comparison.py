# figures/v2_matched_comparison.py
"""
Matched architecture comparison under identical folds.

This script runs 5 configurations under identical 5-fold StratifiedGroupKFold
splits:

  E — XGBoost with 213 segment features            (reference)
  A — XGBoost with 12 SMD descriptors aggregated   (matched with B)
  B — BiLSTM with 12 SMD descriptors per-period    (existing result)
  C — XGBoost with 46 per-period descriptors agg.  (matched with D)
  D — BiLSTM with 46 per-period descriptors        (richer per-period set)

A/B isolates architecture at fixed input (12 SMD).
C/D isolates architecture at fixed input (46 per-period).
A/C isolates feature-count effect for XGBoost.
E shows how much information the per-period representation retains.

A fully matched comparison at 213 features is impossible: 167 of 213
are inherently segment-level (Welch windows spanning multiple cycles,
demographics, period aggregates) and cannot be decomposed per period.
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             balanced_accuracy_score)
from xgboost import XGBClassifier
import h5py

from config import SEED, N_FOLDS, NAN_THRESH, CORR_THRESH
from analysis.v2_utils import load_aurora_raw, RESULTS, OUT_V2
from fold_preprocessor import FoldAwarePreprocessor


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); np.random.seed(SEED)
H5 = os.path.join(OUT_V2, "aurora_unified_v2_sequences.h5")

# ---- The 12 SMD descriptors currently in Table 9 ----
KEEP_12 = ["f_mean", "f_std", "f_max",
           "phi_value", "phi_case", "phi_valid",
           "psi_mean", "psi_std", "psi_max",
           "n_local_maxima", "n_local_minima", "pulse_duration_ms"]


# =========================================================================
# Data loading
# =========================================================================
def load_sequences(h5_path, seg_ids):
    """Load ALL per-period features and the validity mask."""
    with h5py.File(h5_path, "r") as f:
        first = sorted([k for k in f.keys() if k.startswith("segment_")])[0]
        names = [s.decode() if isinstance(s, bytes) else str(s)
                 for s in f[first]["period_feature_names"][()]]
        C = len(names)
        T_max = 0
        # First pass to get T
        sample = f[seg_ids[0]]["period_features"][:]
        T_max = sample.shape[0]
        # Actually T is the MAXLEN stored in HDF5; use the current MAXLEN=40
        # (recompute from the file)
        T_max = 0
        for sid in seg_ids:
            T_max = max(T_max, f[sid]["period_features"].shape[0])
        T_max = min(T_max, 40)  # safety

        seqs = np.zeros((len(seg_ids), T_max, C), dtype=np.float32)
        mask = np.zeros((len(seg_ids), T_max), dtype=bool)
        for i, sid in enumerate(seg_ids):
            pf = f[sid]["period_features"][:]
            n = min(len(pf), T_max)
            seqs[i, :n] = pf[:n]
            mask[i, :n] = True
    return seqs, mask, names


# =========================================================================
# Aggregation for XGBoost
# =========================================================================
def aggregate(seqs, mask):
    """Aggregate per-period sequences to (N, 4*C) via mean/std/max/median."""
    N, T, C = seqs.shape
    agg = np.zeros((N, 4 * C), dtype=np.float32)
    for i in range(N):
        valid = mask[i]
        if valid.sum() == 0:
            continue
        sub = seqs[i][valid]
        agg[i, :C] = sub.mean(0)
        agg[i, C:2*C] = sub.std(0) if len(sub) > 1 else 0.0
        agg[i, 2*C:3*C] = sub.max(0)
        agg[i, 3*C:] = np.median(sub, axis=0)
    return agg


# =========================================================================
# XGBoost CV (5-fold, fold-aware preprocessing for the flat 213 config)
# =========================================================================
def make_xgb():
    return XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0,
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", n_jobs=-1,
        random_state=SEED, verbosity=0)


def cv_xgboost_flat(df, CANDIDATE, y, g, label):
    """For the 213-feature reference: use fold-aware preprocessing."""
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    aucs, f1s, baccs = [], [], []
    f = 0
    for tr, te in sgkf.split(df[CANDIDATE].values, y, groups=g):
        if len(np.unique(y[tr])) < 3 or len(np.unique(y[te])) < 3:
            continue
        pre = FoldAwarePreprocessor(NAN_THRESH, CORR_THRESH).fit(
            df.iloc[tr], CANDIDATE)
        X_tr = pre.transform(df.iloc[tr])
        X_te = pre.transform(df.iloc[te])
        print(f"Fold {f}: {X_tr.shape[1]} features")
        sc = StandardScaler().fit(X_tr)
        m = make_xgb()
        m.fit(sc.transform(X_tr), y[tr])
        prob = m.predict_proba(sc.transform(X_te))
        pred = prob.argmax(1)
        aucs.append(roc_auc_score(y[te], prob, multi_class="ovr", average="macro"))
        f1s.append(f1_score(y[te], pred, average="macro", zero_division=0))
        baccs.append(balanced_accuracy_score(y[te], pred))
        f+=1
    return _summarize(label, aucs, f1s, baccs)


def cv_xgboost_agg(agg, y, g, label):
    """For the aggregated configs: 5-fold with StandardScaler only."""
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    aucs, f1s, baccs = [], [], []
    for tr, te in sgkf.split(agg, y, groups=g):
        if len(np.unique(y[tr])) < 3 or len(np.unique(y[te])) < 3:
            continue
        sc = StandardScaler().fit(agg[tr])
        m = make_xgb()
        m.fit(sc.transform(agg[tr]), y[tr])
        prob = m.predict_proba(sc.transform(agg[te]))
        pred = prob.argmax(1)
        aucs.append(roc_auc_score(y[te], prob, multi_class="ovr", average="macro"))
        f1s.append(f1_score(y[te], pred, average="macro", zero_division=0))
        baccs.append(balanced_accuracy_score(y[te], pred))
    return _summarize(label, aucs, f1s, baccs)


# =========================================================================
# BiLSTM CV
# =========================================================================
class DS(Dataset):
    def __init__(self, X, M, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.M = torch.tensor(M, dtype=torch.bool)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.M[i], self.y[i]


class BiLSTM(nn.Module):
    def __init__(self, d_in, hidden=48, ncls=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(d_in, hidden, num_layers=1, bidirectional=True,
                            batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden * 2, ncls)
    def forward(self, x, m):
        lengths = m.sum(1).clamp(min=1).long().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=x.size(1))
        mm = m.unsqueeze(-1).float()
        h = (out * mm).sum(1) / mm.sum(1).clamp(min=1)
        return self.head(self.drop(h))


def normalize_fold(seq_tr, seq_va, seq_te, m_tr, m_va, m_te):
    """Per-fold normalization with valid-position statistics."""
    def _fit(a, m):
        _, _, C_ = a.shape
        f = a.reshape(-1, C_)
        mflat = m.reshape(-1)
        means = np.zeros(C_); stds = np.ones(C_)
        for j in range(C_):
            col = f[:, j]
            valid = mflat & np.isfinite(col)
            if valid.sum() > 1:
                means[j] = col[valid].mean()
                s = col[valid].std()
                stds[j] = s if s > 1e-6 else 1.0
        return means, stds

    def _apply(a, means, stds, m):
        N_, T_, C_ = a.shape
        f = a.reshape(-1, C_).copy()
        mflat = m.reshape(-1)
        for j in range(C_):
            col = f[:, j]
            valid = mflat & np.isfinite(col)
            f[valid, j] = (col[valid] - means[j]) / stds[j]
            f[~valid, j] = 0.0
        return f.reshape(N_, T_, C_).astype(np.float32)

    m_, s_ = _fit(seq_tr, m_tr)
    return (_apply(seq_tr, m_, s_, m_tr),
            _apply(seq_va, m_, s_, m_va),
            _apply(seq_te, m_, s_, m_te))


def train_bilstm(model, seq_tr, m_tr, y_tr, seq_va, m_va, y_va,
                 epochs=50, lr=5e-4, wd=1e-3, patience=10):
    tr_ld = DataLoader(DS(seq_tr, m_tr, y_tr), batch_size=64, shuffle=True)
    va_ld = DataLoader(DS(seq_va, m_va, y_va), batch_size=256, shuffle=False)

    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()

    best_val, best_state, wait = 0.0, None, 0
    for ep in range(epochs):
        model.train()
        for xb, mb, yb in tr_ld:
            xb, mb, yb = xb.to(DEVICE), mb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb, mb), yb)
            loss.backward(); opt.step()
        model.eval()
        probs, ys = [], []
        with torch.no_grad():
            for xb, mb, yb in va_ld:
                xb, mb = xb.to(DEVICE), mb.to(DEVICE)
                logits = model(xb, mb).cpu().numpy()
                e = np.exp(logits - logits.max(1, keepdims=True))
                probs.append(e / e.sum(1, keepdims=True))
                ys.append(yb.numpy())
        probs = np.vstack(probs); ys = np.concatenate(ys)
        try:
            val_auc = roc_auc_score(ys, probs, multi_class="ovr", average="macro")
        except Exception:
            val_auc = 0.0
        if val_auc > best_val:
            best_val, wait = val_auc, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model


def predict_bilstm(model, seq, m):
    model.eval()
    loader = DataLoader(DS(seq, m, np.zeros(len(seq), dtype=int)),
                        batch_size=256, shuffle=False)
    probs = []
    with torch.no_grad():
        for xb, mb, _ in loader:
            xb, mb = xb.to(DEVICE), mb.to(DEVICE)
            logits = model(xb, mb).cpu().numpy()
            e = np.exp(logits - logits.max(1, keepdims=True))
            probs.append(e / e.sum(1, keepdims=True))
    return np.vstack(probs)


def cv_bilstm(seqs, mask, y, g, label):
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    aucs, f1s, baccs = [], [], []
    C = seqs.shape[2]
    for fold, (tr, te) in enumerate(sgkf.split(seqs, y, groups=g), 1):
        if len(np.unique(y[tr])) < 3 or len(np.unique(y[te])) < 3:
            continue
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15,
                                 random_state=SEED + fold)
        try:
            tr_rel, va_rel = next(gss.split(seqs[tr], y[tr], groups=g[tr]))
        except Exception:
            unique_s = np.unique(g[tr])
            n_val = max(1, len(unique_s) // 6)
            val_s = set(unique_s[-n_val:])
            va_mask = np.array([s in val_s for s in g[tr]])
            tr_rel = np.where(~va_mask)[0]
            va_rel = np.where(va_mask)[0]

        seq_tr, seq_va, seq_te = normalize_fold(
            seqs[tr][tr_rel], seqs[tr][va_rel], seqs[te],
            mask[tr][tr_rel], mask[tr][va_rel], mask[te])

        model = BiLSTM(C)
        model = train_bilstm(model,
                              seq_tr, mask[tr][tr_rel], y[tr][tr_rel],
                              seq_va, mask[tr][va_rel], y[tr][va_rel])
        prob = predict_bilstm(model, seq_te, mask[te])
        pred = prob.argmax(1)
        aucs.append(roc_auc_score(y[te], prob, multi_class="ovr", average="macro"))
        f1s.append(f1_score(y[te], pred, average="macro", zero_division=0))
        baccs.append(balanced_accuracy_score(y[te], pred))
        print(f"    fold {fold}: AUC={aucs[-1]:.4f}")
    return _summarize(label, aucs, f1s, baccs)


def _summarize(label, aucs, f1s, baccs):
    return {
        "configuration": label,
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs, ddof=1)),
        "f1_mean": float(np.mean(f1s)),
        "bacc_mean": float(np.mean(baccs)),
        "n_folds": len(aucs),
    }


# =========================================================================
# Main
# =========================================================================
def main():
    print("=" * 72)
    print("Matched architecture comparison — 5 configurations, same folds")
    print("=" * 72)
    print(f"Device: {DEVICE}")

    # Load flat data
    aur, CANDIDATE = load_aurora_raw()
    y = aur.htn3.values
    g = aur.subject_id.astype(str).values
    seg_ids = aur.segment_id.values
    print(f"\nAurora: {len(aur)} segments, {aur.subject_id.nunique()} subjects, "
          f"{len(CANDIDATE)} candidates")

    # Load sequences (all 46 features)
    print(f"\nLoading sequences from {H5}")
    seqs_all, mask, names = load_sequences(H5, seg_ids)
    print(f"  Sequence tensor: {seqs_all.shape}  ({len(names)} features)")
    print(f"  Feature names: {names}")

    # Check which features are non-degenerate
    valid_idx = np.where(mask)
    all_vals = seqs_all[valid_idx[0], valid_idx[1], :]
    stds = all_vals.std(axis=0)
    print(f"\n  Feature stds (min/max): {stds.min():.6f} / {stds.max():.6f}")
    n_const = int((stds < 1e-6).sum())
    if n_const > 0:
        print(f"  WARNING: {n_const} features have near-zero variance across valid positions")
        for i in np.where(stds < 1e-6)[0]:
            print(f"    {names[i]}: std={stds[i]:.6f}")

    # Slice to the 12 SMD subset
    idx_12 = [i for i, n in enumerate(names) if n in KEEP_12]
    print(f"\n  12 SMD subset indices: {idx_12} ({len(idx_12)} features)")
    seqs_12 = seqs_all[:, :, idx_12]
    assert seqs_12.shape[2] == 12, f"Expected 12, got {seqs_12.shape[2]}"

    results = []

    # ---- E: XGBoost with 213 features (reference) ----
    print("\n[E] XGBoost with 213 segment features (reference)")
    r_e = cv_xgboost_flat(aur, CANDIDATE, y, g, "E: XGB-213 (reference)")
    results.append(r_e)
    print(f"    AUC = {r_e['auc_mean']:.4f} ± {r_e['auc_std']:.4f}")

    # ---- A: XGBoost with 12 SMD aggregated ----
    print("\n[A] XGBoost with 12 SMD descriptors (aggregated)")
    agg_12 = aggregate(seqs_12, mask)
    print(f"    Aggregated feature count: {agg_12.shape[1]}")
    r_a = cv_xgboost_agg(agg_12, y, g, "A: XGB-12 (agg)")
    results.append(r_a)
    print(f"    AUC = {r_a['auc_mean']:.4f} ± {r_a['auc_std']:.4f}")

    # ---- C: XGBoost with 46 per-period aggregated ----
    print("\n[C] XGBoost with 46 per-period descriptors (aggregated)")
    agg_46 = aggregate(seqs_all, mask)
    print(f"    Aggregated feature count: {agg_46.shape[1]}")
    r_c = cv_xgboost_agg(agg_46, y, g, "C: XGB-46 (agg)")
    results.append(r_c)
    print(f"    AUC = {r_c['auc_mean']:.4f} ± {r_c['auc_std']:.4f}")

    # ---- B: BiLSTM with 12 SMD (existing result, re-run for consistency) ----
    print("\n[B] BiLSTM with 12 SMD descriptors (per-period sequence)")
    r_b = cv_bilstm(seqs_12, mask, y, g, "B: BiLSTM-12")
    results.append(r_b)
    print(f"    AUC = {r_b['auc_mean']:.4f} ± {r_b['auc_std']:.4f}")

    # ---- D: BiLSTM with 46 per-period ----
    print("\n[D] BiLSTM with 46 per-period descriptors (per-period sequence)")
    r_d = cv_bilstm(seqs_all, mask, y, g, "D: BiLSTM-46")
    results.append(r_d)
    print(f"    AUC = {r_d['auc_mean']:.4f} ± {r_d['auc_std']:.4f}")

    # ---- Summary ----
    df = pd.DataFrame(results)
    out = os.path.join(RESULTS, "v2_matched_comparison.csv")
    df.to_csv(out, index=False)
    print("\n" + "=" * 72)
    print("MATCHED ARCHITECTURE COMPARISON — SUMMARY")
    print("=" * 72)
    print(df.round(4).to_string(index=False))
    print(f"\nSaved: {out}")

    # ---- Print the key comparisons ----
    print("\n" + "=" * 72)
    print("KEY COMPARISONS")
    print("=" * 72)
    dict_results = {r["configuration"].split(":")[0].strip(): r for r in results}
    a = dict_results["A"]
    b = dict_results["B"]
    c = dict_results["C"]
    d = dict_results["D"]
    e = dict_results["E"]

    print(f"\nArchitecture effect at 12 features:")
    print(f"  XGB-12:    {a['auc_mean']:.4f} ± {a['auc_std']:.4f}")
    print(f"  BiLSTM-12: {b['auc_mean']:.4f} ± {b['auc_std']:.4f}")
    print(f"  Δ (Seq − Flat) = {b['auc_mean'] - a['auc_mean']:+.4f}")

    print(f"\nArchitecture effect at 46 per-period features:")
    print(f"  XGB-46:    {c['auc_mean']:.4f} ± {c['auc_std']:.4f}")
    print(f"  BiLSTM-46: {d['auc_mean']:.4f} ± {d['auc_std']:.4f}")
    print(f"  Δ (Seq − Flat) = {d['auc_mean'] - c['auc_mean']:+.4f}")

    print(f"\nFeature-count effect for XGBoost:")
    print(f"  XGB-12  → XGB-46  : {c['auc_mean'] - a['auc_mean']:+.4f}")
    print(f"  XGB-46  → XGB-213 : {e['auc_mean'] - c['auc_mean']:+.4f}")

    print(f"\nInformation retained by per-period features:")
    print(f"  213-feature XGB (reference): {e['auc_mean']:.4f}")
    print(f"  46-feature XGB (aggregated): {c['auc_mean']:.4f}")


if __name__ == "__main__":
    main()
