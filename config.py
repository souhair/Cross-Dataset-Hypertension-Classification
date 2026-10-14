# config.py
"""Centralized configuration for all scripts."""

SEED = 42
N_FOLDS = 5

# ============================================================
# XGBoost configurations
# ============================================================
XGB_3CLASS = dict(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0,
    objective="multi:softprob", num_class=3,
    eval_metric="mlogloss", n_jobs=-1,
    random_state=SEED, verbosity=0,
)

XGB_BINARY = dict(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0,
    objective="binary:logistic", eval_metric="logloss",
    n_jobs=-1, random_state=SEED, verbosity=0,
)

XGB_FINETUNE = dict(
    n_estimators=200, max_depth=4, learning_rate=0.02,
    subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, reg_alpha=0.0,
    objective="multi:softprob", num_class=3,
    eval_metric="mlogloss", n_jobs=-1,
    random_state=SEED, verbosity=0,
)

XGB_FINETUNE_BINARY = dict(
    n_estimators=200, max_depth=4, learning_rate=0.02,
    subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, reg_alpha=0.0,
    objective="binary:logistic", eval_metric="logloss",
    n_jobs=-1, random_state=SEED, verbosity=0,
)

RF_PARAMS = dict(
    n_estimators=600, max_depth=12, min_samples_leaf=4,
    class_weight="balanced", n_jobs=-1, random_state=SEED,
)

LR_PARAMS = dict(max_iter=4000, C=0.5, class_weight="balanced")

# ============================================================
# Best XGBoost params from Optuna search
# ============================================================
XGB_BEST = dict(
    n_estimators=200, max_depth=4,
    learning_rate=0.0661987577743696,
    subsample=0.6300160837427259,
    colsample_bytree=0.933663830490076,
    reg_lambda=0.5660157298534864,
    reg_alpha=1.9346735110920752,
    objective="multi:softprob", num_class=3,
    eval_metric="mlogloss", n_jobs=-1,
    random_state=SEED, verbosity=0,
)

# ============================================================
# Fold-aware preprocessing
# ============================================================
NAN_THRESH = 0.5
CORR_THRESH = 0.95

# ============================================================
# External validation
# ============================================================
FT_FRACTION = 0.70   # 70% of target subjects for fine-tuning, 30% for test
EXTERNAL_SEED = SEED