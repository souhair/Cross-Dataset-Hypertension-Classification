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


# run_pulsedb_vital_full_v2.py
import logging, os, time, multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

from extractors.ppg_feature_extractor import (
    process_mat_file_unified, flatten_features,
    save_batch_csv, save_batch_sequences, PERIOD_P_DEFAULT,
    prepare_output,
)

ROOT = r"pulsedb_vital"
OUT_DIR = r"add_your_root_folder\out_v2"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, "pulsedb_vital_unified_v2.csv")
OUT_H5 = os.path.join(OUT_DIR, "pulsedb_vital_unified_v2_sequences.h5")
PER_CSV = os.path.join(OUT_DIR, "pulsedb_vital_unified_v2_periods.csv")
FS, WORKERS, SAVE_EVERY = 125, 6, 500


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    prepare_output(OUT_CSV, OUT_H5, overwrite=True)

    mat_files = sorted([f for f in os.listdir(ROOT) if f.endswith(".mat")])
    print(f"[run] files: {len(mat_files)}")

    batch_csv, batch_seq, batch_files = [], [], []
    seg_idx = ok = rej = err = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(process_mat_file_unified,
                          os.path.join(ROOT, fname),
                          None, FS, PERIOD_P_DEFAULT): fname
                for fname in mat_files}
        for i, fut in enumerate(as_completed(futs), 1):
            fname = futs[fut]
            try:
                feats = fut.result()
            except Exception as e:
                err += 1; logging.error(f"{fname}: {e}"); continue
            if not feats:
                rej += 1; continue
            ok += 1
            batch_csv.extend(flatten_features(feats))
            batch_seq.extend([f["seq"] for f in feats])
            batch_files.append(fname)
            if len(batch_csv) >= SAVE_EVERY:
                save_batch_csv(batch_csv, OUT_CSV, OUT_CSV + ".log",
                                batch_files, seg_idx)
                seg_idx = save_batch_sequences(
                    batch_seq, OUT_H5, OUT_H5 + ".log",
                    batch_files, seg_idx, period_output_csv=PER_CSV)
                batch_csv, batch_seq, batch_files = [], [], []
            if i % 25 == 0:
                el = time.time() - t0
                logging.info(f"{i}/{len(mat_files)}  ok={ok} rej={rej} err={err}  "
                             f"ETA={(len(mat_files)-i)/(i/el)/60:.1f} min")

    if batch_csv:
        save_batch_csv(batch_csv, OUT_CSV, OUT_CSV + ".log",
                        batch_files, seg_idx)
        save_batch_sequences(batch_seq, OUT_H5, OUT_H5 + ".log",
                              batch_files, seg_idx, period_output_csv=PER_CSV)
    print(f"\nDONE  ok={ok}  rej={rej}  err={err}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()