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


# run_ppgbp_full_v2.py
import logging, os, time, multiprocessing

from adapters.ppgbp_adapter import load_ppgbp_segments
from extractors.ppg_feature_extractor import (
    process_single_segment_unified, save_batch_csv, save_batch_sequences,
    PERIOD_P_DEFAULT, prepare_output,
)

ROOT = (r"add_your_root_folder"
        r"\5459299\PPG-BP Database\Data File")
OUT_DIR = r"add_your_root_folder\out_v2"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, "ppgbp_unified_v2.csv")
OUT_H5 = os.path.join(OUT_DIR, "ppgbp_unified_v2_sequences.h5")
PER_CSV = os.path.join(OUT_DIR, "ppgbp_unified_v2_periods.csv")
FS, PERIOD_P = 125, PERIOD_P_DEFAULT


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    prepare_output(OUT_CSV, OUT_H5, overwrite=True)

    segs = load_ppgbp_segments(ROOT, target_fs=FS)
    print(f"[run] segments: {len(segs)}")

    batch_csv, batch_seq, batch_files = [], [], []
    ok = rej = err = 0
    t0 = time.time()

    for i, seg in enumerate(segs, 1):
        try:
            out = process_single_segment_unified(
                seg, seg["source_file"], fs=FS, period_p=PERIOD_P)
        except Exception as e:
            err += 1; logging.error(f"{seg['source_file']}: {e}"); continue
        if out is None:
            rej += 1; continue
        ok += 1
        batch_csv.append(out["flat"])
        batch_seq.append(out["seq"])
        batch_files.append(seg["source_file"])

    if batch_csv:
        save_batch_csv(batch_csv, OUT_CSV, OUT_CSV + ".log", batch_files, 0)
        save_batch_sequences(batch_seq, OUT_H5, OUT_H5 + ".log",
                             batch_files, 0, period_output_csv=PER_CSV)
    print(f"\nDONE  ok={ok}  rej={rej}  err={err}  "
          f"time={(time.time()-t0)/60:.2f} min")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()