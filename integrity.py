# integrity.py
"""Reusable assertions for extraction and evaluation scripts."""


def assert_extraction_valid(df, expected_min_subjects=100):
    assert len(df) > 0, "Empty dataframe"
    assert df.subject_id.nunique() >= expected_min_subjects, \
        f"Only {df.subject_id.nunique()} subjects"
    assert df.subject_id.isna().sum() == 0, "NaN subject IDs"
    assert (df.subject_id.astype(str) != "0").all(), \
        "Subject IDs collapsed to 0"
    print(f"✓ Extraction valid: {df.subject_id.nunique()} subjects, "
          f"{len(df)} segments")


def assert_no_split_overlap(train_subjects, test_subjects):
    overlap = set(train_subjects) & set(test_subjects)
    assert len(overlap) == 0, f"Overlap: {list(overlap)[:5]}"
    assert len(train_subjects) > 0 and len(test_subjects) > 0
    print(f"✓ No overlap: {len(train_subjects)} train / "
          f"{len(test_subjects)} test")


def assert_no_nan(X, name="matrix"):
    import numpy as np
    assert not np.isnan(X).any(), f"NaN in {name}"
    assert not np.isinf(X).any(), f"inf in {name}"