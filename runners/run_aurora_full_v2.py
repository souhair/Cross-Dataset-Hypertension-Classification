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


# run_aurora_full_v2.py
import logging, os, time, multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

from adapters.aurora_tsv_adapter_fixed import load_aurora_segments
from extractors.ppg_feature_extractor import (
    process_single_segment_unified, save_batch_csv, save_batch_sequences,
    PERIOD_P_DEFAULT, prepare_output,
)

ROOT = r"Aurora"
OUT_DIR = r"add_your_root_folder\out_v2"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, "aurora_unified_v2.csv")
OUT_H5 = os.path.join(OUT_DIR, "aurora_unified_v2_sequences.h5")
PER_CSV = os.path.join(OUT_DIR, "aurora_unified_v2_periods.csv")
LOG_CSV = OUT_CSV + ".log"
LOG_H5 = OUT_H5 + ".log"
FS, WORKERS, SAVE_EVERY = 125, 6, 500


def _worker(payload):
    seg, fs, p = payload
    try:
        out = process_single_segment_unified(
            seg, seg["source_file"], fs=fs, period_p=p)
        return seg, out, None
    except Exception as e:
        return seg, None, repr(e)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    prepare_output(OUT_CSV, OUT_H5, overwrite=True)

    segs = load_aurora_segments(ROOT, protocol="auscultatory",
                                 exclude_calibration=True, quality_min=0.5,
                                 min_duration_s=10.0, target_fs=FS)
    print(f"[run] segments: {len(segs)}")

    batch_csv, batch_seq, batch_files = [], [], []
    seg_idx = ok = rej = err = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(_worker, (s, FS, PERIOD_P_DEFAULT)) for s in segs]
        for i, fut in enumerate(as_completed(futs), 1):
            seg, out, e = fut.result()
            if e is not None:
                err += 1; logging.error(f"{seg['SubjectID']}: {e}"); continue
            if out is None:
                rej += 1; continue
            ok += 1
            batch_csv.append(out["flat"])
            batch_seq.append(out["seq"])
            batch_files.append(seg["source_file"])
            if len(batch_csv) >= SAVE_EVERY:
                save_batch_csv(batch_csv, OUT_CSV, LOG_CSV, batch_files, seg_idx)
                seg_idx = save_batch_sequences(
                    batch_seq, OUT_H5, LOG_H5, batch_files, seg_idx,
                    period_output_csv=PER_CSV)
                batch_csv, batch_seq, batch_files = [], [], []
            if i % 200 == 0:
                el = time.time() - t0
                logging.info(f"{i}/{len(segs)}  ok={ok} rej={rej} err={err}  "
                             f"ETA={(len(segs)-i)/(i/el)/60:.1f} min")

    if batch_csv:
        save_batch_csv(batch_csv, OUT_CSV, LOG_CSV, batch_files, seg_idx)
        save_batch_sequences(batch_seq, OUT_H5, LOG_H5, batch_files, seg_idx,
                             period_output_csv=PER_CSV)
    print(f"\nDONE  ok={ok}  rej={rej}  err={err}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()