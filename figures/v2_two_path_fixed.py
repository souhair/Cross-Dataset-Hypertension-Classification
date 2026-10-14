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


# v2_two_path_fixed.py  (final)
"""
Flat vs sequence comparison on Aurora-BP .

  1. Uses existing load_aurora_raw() (no missing helper).
  2. 5-fold StratifiedGroupKFold instead of single 80/20 split.
  3. Fold-aware preprocessing (FoldAwarePreprocessor) inside each fold.
  4. Sequence-model early stopping on a subject-grouped inner split
     drawn from the training fold, never the test fold.
  5. Per-fold normalization for the sequence array.
  6. Segments matched explicitly by segment_id between the flat CSV and
     the HDF5 sequence file (no reliance on row order).
"""
import os, warnings
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
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import h5py

from config import SEED, N_FOLDS, NAN_THRESH, CORR_THRESH
from analysis.v2_utils import load_aurora_raw, RESULTS, OUT_V2
from fold_preprocessor import FoldAwarePreprocessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); np.random.seed(SEED)

H5 = os.path.join(OUT_V2, "aurora_unified_v2_sequences.h5")

# ---- Load flat features --------------------------------------------------
aur, CANDIDATE = load_aurora_raw()
y = aur.htn3.values
g = aur.subject_id.astype(str).values
X_flat = aur[CANDIDATE].values

print(f"Loaded flat: {aur.shape[0]} rows  |  {len(CANDIDATE)} candidate features")
print(f"Subjects: {aur.subject_id.nunique()}   Classes: {np.bincount(y)}")

# ---- Load sequence features (match by segment_id) ------------------------
KEEP = ["f_mean", "f_std", "f_max", "phi_value", "phi_case", "phi_valid",
        "psi_mean", "psi_std", "psi_max", "n_local_maxima", "n_local_minima",
        "pulse_duration_ms"]
MAXLEN = 40

seg_ids = aur.segment_id.values

with h5py.File(H5, "r") as f:
    h5_keys = set(k for k in f.keys() if k.startswith("segment_"))
    missing = [s for s in seg_ids if s not in h5_keys]
    if missing:
        raise SystemExit(
            f"{len(missing)} segment_ids from the CSV are missing in {H5}. "
            f"First missing: {missing[:3]}. Re-run the Aurora extraction "
            f"so both files come from the same run."
        )

    first = sorted(h5_keys)[0]
    names = [s.decode() if isinstance(s, bytes) else str(s)
             for s in f[first]["period_feature_names"][()]]
    keep_idx = [i for i, n in enumerate(names) if n in KEEP]
    C = len(keep_idx)
    print(f"Sequence: {len(names)} period features, "
          f"{len(keep_idx)} kept ({C} descriptors)")

    seqs = np.zeros((len(seg_ids), MAXLEN, C), dtype=np.float32)
    mask = np.zeros((len(seg_ids), MAXLEN), dtype=bool)
    for i, sid in enumerate(seg_ids):
        pf = f[sid]["period_features"][:][:, keep_idx]
        n = min(len(pf), MAXLEN)
        seqs[i, :n] = pf[:n]
        mask[i, :n] = True

print(f"Sequence tensor: {seqs.shape}\n")


class DS(Dataset):
    def __init__(self, X, M, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.M = torch.tensor(M, dtype=torch.bool)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.M[i], self.y[i]


class Transformer(nn.Module):
    def __init__(self, d_in, d_model=32, nhead=2, nlayers=1, ncls=3, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=64,
                                           dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, nlayers)
        self.head = nn.Linear(d_model, ncls)
    def forward(self, x, m):
        h = self.proj(x)
        h = self.enc(h, src_key_padding_mask=~m)
        mm = m.unsqueeze(-1).float()
        h = (h * mm).sum(1) / mm.sum(1).clamp(min=1)
        return self.head(h)


class BiLSTM(nn.Module):
    def __init__(self, d_in, hidden=48, ncls=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(d_in, hidden, num_layers=1, bidirectional=True,
                            batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden * 2, ncls)

    def forward(self, x, m):
        # Packed sequence — LSTM never sees padding positions.
        lengths = m.sum(1).clamp(min=1).long().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=x.size(1))
        mm = m.unsqueeze(-1).float()
        h = (out * mm).sum(1) / mm.sum(1).clamp(min=1)
        return self.head(self.drop(h))


def normalize_fold(seq_tr, seq_va, seq_te, mask_tr=None, mask_va=None, mask_te=None):
    """Per-fold normalization with validity masks.

    Only finite, valid (non-padded) entries contribute to the mean/std.
    Imputed / padded entries are set to 0 in standardized coordinates.
    """
    def _fit(a, m):
        _, _, C_ = a.shape
        f = a.reshape(-1, C_)
        mflat = m.reshape(-1) if m is not None else np.ones(f.shape[0], bool)
        means = np.zeros(C_); stds = np.ones(C_)
        for j in range(C_):
            col = f[:, j]
            valid = mflat & np.isfinite(col)
            if valid.sum() > 1:
                means[j] = col[valid].mean()
                stds[j] = col[valid].std() + 1e-10
        return means, stds

    def _apply(a, means, stds, m):
        N_, T_, C_ = a.shape
        f = a.reshape(-1, C_).copy()
        mflat = m.reshape(-1) if m is not None else np.ones(f.shape[0], bool)
        for j in range(C_):
            col = f[:, j]
            valid = mflat & np.isfinite(col)
            # Valid entries: standardize
            f[valid, j] = (col[valid] - means[j]) / stds[j]
            # Invalid / imputed / padded entries: exactly zero
            f[~valid, j] = 0.0
        return f.reshape(N_, T_, C_).astype(np.float32)

    m_, s_ = _fit(seq_tr, mask_tr)
    return (_apply(seq_tr, m_, s_, mask_tr),
            _apply(seq_va, m_, s_, mask_va),
            _apply(seq_te, m_, s_, mask_te))


def train_eval_seq(name, model_fn, seq_tr, m_tr, y_tr,
                    seq_va, m_va, y_va, seq_te, m_te, y_te,
                    epochs=50, lr=5e-4, wd=1e-3, patience=10):
    tr_ld = DataLoader(DS(seq_tr, m_tr, y_tr), batch_size=64, shuffle=True)
    va_ld = DataLoader(DS(seq_va, m_va, y_va), batch_size=256, shuffle=False)
    te_ld = DataLoader(DS(seq_te, m_te, y_te), batch_size=256, shuffle=False)

    model = model_fn().to(DEVICE)
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
    model.eval()

    probs, ys = [], []
    with torch.no_grad():
        for xb, mb, yb in te_ld:
            xb, mb = xb.to(DEVICE), mb.to(DEVICE)
            logits = model(xb, mb).cpu().numpy()
            e = np.exp(logits - logits.max(1, keepdims=True))
            probs.append(e / e.sum(1, keepdims=True))
            ys.append(yb.numpy())
    probs = np.vstack(probs); ys = np.concatenate(ys)
    pred = probs.argmax(1)

    return {
        "auc": roc_auc_score(ys, probs, multi_class="ovr", average="macro"),
        "macro_f1": f1_score(ys, pred, average="macro", zero_division=0),
        "bacc": balanced_accuracy_score(ys, pred),
        "acc": accuracy_score(ys, pred),
    }


def main():
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    flat_results = {"LogReg": [], "XGB": []}
    seq_results = {"Transformer": [], "BiLSTM": []}

    for fold, (tr, te) in enumerate(sgkf.split(X_flat, y, groups=g), 1):
        if len(np.unique(y[tr])) < 3 or len(np.unique(y[te])) < 3:
            continue
        print(f"--- Fold {fold}  (train={len(tr)}, test={len(te)}) ---")

        # ==============================================================
        # Path A — flat, fold-aware preprocessing
        # ==============================================================
        pre = FoldAwarePreprocessor(NAN_THRESH, CORR_THRESH).fit(
            aur.iloc[tr], CANDIDATE)
        X_tr = pre.transform(aur.iloc[tr])
        X_te = pre.transform(aur.iloc[te])
        sc = StandardScaler().fit(X_tr)
        Xtr, Xte = sc.transform(X_tr), sc.transform(X_te)

        for name, m in [
            ("LogReg", LogisticRegression(max_iter=4000, C=0.5,
                                           class_weight="balanced")),
            ("XGB", XGBClassifier(
                n_estimators=500, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0,
                objective="multi:softprob", num_class=3,
                eval_metric="mlogloss", n_jobs=-1, random_state=SEED,
                verbosity=0)),
        ]:
            m.fit(Xtr, y[tr])
            prob = m.predict_proba(Xte)
            pred = prob.argmax(1)
            flat_results[name].append({
                "auc": roc_auc_score(y[te], prob, multi_class="ovr",
                                      average="macro"),
                "f1": f1_score(y[te], pred, average="macro", zero_division=0),
                "acc": accuracy_score(y[te], pred),
                "bacc": balanced_accuracy_score(y[te], pred),
            })
            print(f"  Flat {name:6s}  "
                  f"AUC={flat_results[name][-1]['auc']:.4f}  "
                  f"F1={flat_results[name][-1]['f1']:.4f}")

        # ==============================================================
        # Path B — sequence, subject-grouped inner split for early stopping
        # ==============================================================
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
            mask_tr=mask[tr][tr_rel], mask_va=mask[tr][va_rel],
            mask_te=mask[te])

        for name, fn in [
            ("Transformer", lambda: Transformer(C)),
            ("BiLSTM", lambda: BiLSTM(C)),
        ]:
            r = train_eval_seq(name, fn,
                                seq_tr, mask[tr][tr_rel], y[tr][tr_rel],
                                seq_va, mask[tr][va_rel], y[tr][va_rel],
                                seq_te, mask[te], y[te])
            seq_results[name].append(r)
            print(f"  Seq  {name:12s}  "
                  f"AUC={r['auc']:.4f}  F1={r['macro_f1']:.4f}")

    # ==============================================================
    # Summary
    # ==============================================================
    rows = []
    for path, name in [("Flat", "LogReg"), ("Flat", "XGB"),
                       ("Sequence", "Transformer"), ("Sequence", "BiLSTM")]:
        results = flat_results[name] if path == "Flat" else seq_results[name]
        if not results:
            continue
        df = pd.DataFrame(results)
        f1_col = "f1" if "f1" in df.columns else "macro_f1"
        rows.append({
            "path": path, "model": name,
            "auc_mean": df["auc"].mean(), "auc_std": df["auc"].std(),
            "f1_mean": df[f1_col].mean(),
            "acc_mean": df["acc"].mean(),
            "bacc_mean": df["bacc"].mean(),
        })
    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(os.path.join(RESULTS, "v2_two_path_corrected.csv"),
                       index=False)
    print("\n" + "=" * 72)
    print("FLAT vs SEQUENCE — corrected, 5-fold StratifiedGroupKFold")
    print("=" * 72)
    print(df_summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()