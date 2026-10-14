# fold_preprocessor.py
"""Training-fold-only preprocessing pipeline.
"""
import numpy as np
import pandas as pd


class FoldAwarePreprocessor:
    """
    Fit imputation, correlation filter, and feature selection on a
    training fold only. Apply the same transforms to test.
    """
    def __init__(self, nan_thresh=0.5, corr_thresh=0.95):
        self.nan_thresh = nan_thresh
        self.corr_thresh = corr_thresh
        self.medians_ = None
        self.feature_list_ = None

    def fit(self, df_train, candidate_features):
        # 0. Replace ±inf with NaN BEFORE any statistic is computed
        train = df_train[candidate_features].replace([np.inf, -np.inf], np.nan)

        # 1. Missingness screening on TRAINING rows only
        nan_frac = train.isna().mean()
        keep = nan_frac[nan_frac <= self.nan_thresh].index.tolist()

        # 2. Median imputation on TRAINING rows only (now guaranteed finite)
        self.medians_ = {
            c: float(train[c].median()) if train[c].notna().any() else 0.0
            for c in keep
        }

        # 3. Correlation filter on TRAINING rows only
        X = train[keep].copy()
        for c in keep:
            X[c] = X[c].fillna(self.medians_[c])
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = set()
        for (a, b), r in upper.stack().items():
            if r > self.corr_thresh:
                to_drop.add(b if a < b else a)

        self.feature_list_ = [c for c in keep if c not in to_drop]
        return self

    def transform(self, df):
        X = df[self.feature_list_].replace([np.inf, -np.inf], np.nan)
        for c in self.feature_list_:
            X[c] = X[c].fillna(self.medians_[c])
        return X.values
