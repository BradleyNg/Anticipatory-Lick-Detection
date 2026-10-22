#!/usr/bin/env python3
"""
compare_ground_truth.py
=======================

Score the detected anticipatory licks against a hand-scored ground-truth file.

The hand-scored format is not fixed yet, so this auto-detects:
  * which column holds the timestamps (or a start/end pair -> "period" mode);
  * the units -- seconds, milliseconds, or frame numbers -- by checking which
    interpretation puts the values inside the video's actual duration.

What it reports
---------------
* precision / recall / F1 of detected anticipatory licks vs the hand scores;
* the median and IQR of the signed time offset of matched pairs -- a non-zero median
  means a systematic clock shift, not a detection problem;
* for every hand-scored lick we did NOT match, whether it landed inside a detected
  CONSUMMATORY or OTHER bout. That separates "we never saw the lick" (a detection miss)
  from "we saw it but filed it under a different class" (a classification error) --
  which is the question that actually matters for the filtering.

Usage
-----
    python compare_ground_truth.py GROUND_TRUTH.csv \
        --licks <stem>_anticipatory_licks.csv \
        --bouts <stem>_lick_bouts.csv

Hand scoring that marks only where licking starts (one mark for a single lick or a whole bout)
is matched against bout starts instead:

    python compare_ground_truth.py GROUND_TRUTH.csv \
        --licks <stem>_anticipatory_bouts.csv --detect-time-col start_s \
        --bouts <stem>_lick_bouts.csv

Add --offset-scan if the ground truth may be on a different clock than the video.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

TIME_HINTS = ("time", "timestamp", "onset", "start", "sec", "ms", "frame", "t")
END_HINTS = ("end", "offset", "stop")


# --------------------------------------------------------------------------------------
# Loading / auto-detection
# --------------------------------------------------------------------------------------
def _numeric_cols(df):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def pick_time_columns(df, time_col=None, end_col=None):
    """Return (start_col, end_col_or_None)."""
    if time_col:
        return time_col, end_col
    nums = _numeric_cols(df)
    if not nums:
        raise SystemExit(f"no numeric column in ground truth; columns={list(df.columns)}")

    def score(c):
        lc = str(c).lower()
        return max((len(h) for h in TIME_HINTS if h in lc), default=0)

    ends = [c for c in nums if any(h in str(c).lower() for h in END_HINTS)]
    starts = [c for c in nums if c not in ends]
    if not starts:
        starts, ends = nums, []
    start = max(starts, key=score)
    end = max(ends, key=score) if ends else None
    return start, end


def detect_units(values, duration_s, n_frames, forced="auto"):
    """Decide whether `values` are seconds, milliseconds or frame numbers."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return "s", 1.0
    if forced != "auto":
        return forced, {"s": 1.0, "ms": 1e-3, "frames": None}[forced]

    cands = {"s": 1.0, "ms": 1e-3, "frames": 1.0 / max(n_frames, 1) * duration_s}
    best, best_score = "s", -1.0
    for unit, k in cands.items():
        secs = v * k
        inside = float(np.mean((secs >= -1.0) & (secs <= duration_s * 1.05)))
        # prefer interpretations that both fit inside AND use a decent span of the video
        span = float(np.ptp(secs)) / duration_s if duration_s > 0 else 0.0
        s = inside + 0.25 * min(span, 1.0)
        if s > best_score:
            best, best_score = unit, s
    return best, cands[best]


# --------------------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------------------
def greedy_match(gt, det, tol):
    """One-to-one nearest matching within `tol`. Returns (pairs, gt_unmatched, det_unmatched).
    pairs is a list of (gt_index, det_index, signed_offset = det - gt)."""
    pairs = []
    if len(gt) and len(det):
        cand = []
        order = np.argsort(det)
        det_sorted = np.asarray(det)[order]
        for gi, t in enumerate(gt):
            lo = np.searchsorted(det_sorted, t - tol, "left")
            hi = np.searchsorted(det_sorted, t + tol, "right")
            for k in range(lo, hi):
                di = int(order[k])
                cand.append((abs(det[di] - t), gi, di))
        cand.sort()
        used_g, used_d = set(), set()
        for _, gi, di in cand:
            if gi in used_g or di in used_d:
                continue
            used_g.add(gi)
            used_d.add(di)
            pairs.append((gi, di, det[di] - gt[gi]))
    matched_g = {p[0] for p in pairs}
    matched_d = {p[1] for p in pairs}
    return (pairs,
            [i for i in range(len(gt)) if i not in matched_g],
            [i for i in range(len(det)) if i not in matched_d])


def overlap_match(gt_periods, bouts):
    """Period-mode: a GT period counts as detected if it overlaps an anticipatory bout."""
    hits, misses = 0, []
    for i, (a, b) in enumerate(gt_periods):
        if any((a < e) and (b > s) for s, e in bouts):
            hits += 1
        else:
            misses.append(i)
    return hits, misses


# --------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Score detected anticipatory licks vs hand scoring.")
    ap.add_argument("ground_truth", help="hand-scored CSV of anticipatory lick timestamps")
    ap.add_argument("--licks", required=True, help="<stem>_anticipatory_licks.csv")
    ap.add_argument("--bouts", default=None,
                    help="<stem>_lick_bouts.csv (enables the misclassified-vs-missed diagnostic)")
    ap.add_argument("--fps", type=float, default=30.0,
                    help="video frame rate, used when the hand scores are frame numbers (default 30)")
    ap.add_argument("--gt-time-col", default=None)
    ap.add_argument("--gt-end-col", default=None)
    ap.add_argument("--gt-units", default="auto", choices=["auto", "s", "ms", "frames"])
    ap.add_argument("--detect-time-col", default="onset_s", choices=["onset_s", "peak_s", "start_s"],
                    help="which detected timestamp to match on (default: protrusion onset). Use "
                         "start_s with --licks <stem>_anticipatory_bouts.csv when the hand scoring "
                         "marks where licking starts (a single lick or a bout) rather than every lick")
    ap.add_argument("--match-tolerance", type=float, default=0.25,
                    help="seconds; a hand score within this of a detection counts as matched")
    ap.add_argument("--offset-scan", action="store_true",
                    help="search for the clock offset that maximises matches")
    ap.add_argument("--offset-range", type=float, default=5.0, help="seconds, +/-")
    ap.add_argument("--offset-step", type=float, default=0.01)
    ap.add_argument("--out", default=None, help="write per-hand-score match results to this CSV")
    args = ap.parse_args(argv)

    det_df = pd.read_csv(args.licks)
    if args.detect_time_col not in det_df.columns:
        raise SystemExit(f"{args.licks} has no {args.detect_time_col}; "
                         f"columns={list(det_df.columns)}")
    det = np.sort(det_df[args.detect_time_col].to_numpy(float))

    bouts = pd.read_csv(args.bouts) if args.bouts else None
    if bouts is not None and len(bouts):
        duration_s = float(bouts["end_s"].max())
        n_frames = int(bouts["end_frame"].max())
    else:
        duration_s = float(det.max()) if len(det) else 1.0
        n_frames = int(duration_s * args.fps)

    gt_df = pd.read_csv(args.ground_truth)
    scol, ecol = pick_time_columns(gt_df, args.gt_time_col, args.gt_end_col)
    unit, k = detect_units(gt_df[scol], duration_s, n_frames, args.gt_units)
    if unit == "frames":
        k = 1.0 / args.fps
    gt = gt_df[scol].to_numpy(float) * k
    print(f"ground truth : {args.ground_truth}")
    print(f"  time column : {scol!r}" + (f" (+ end {ecol!r})" if ecol else ""))
    print(f"  units       : {unit}  ->  range {gt.min():.2f}-{gt.max():.2f}s "
          f"(video is ~{duration_s:.1f}s)")
    print(f"  rows        : {len(gt)}")
    what = "bouts" if args.detect_time_col == "start_s" else "licks"
    print(f"detected     : {len(det)} anticipatory {what} from {args.licks}")

    # ---------------- period mode ----------------
    if ecol is not None:
        gt_end = gt_df[ecol].to_numpy(float) * k
        periods = list(zip(gt, gt_end))
        ab = bouts[bouts["classification"] == "anticipatory"] if bouts is not None else None
        if ab is None or not len(ab):
            raise SystemExit("period mode needs --bouts with anticipatory rows")
        bl = list(zip(ab["start_s"].to_numpy(float), ab["end_s"].to_numpy(float)))
        hits, misses = overlap_match(periods, bl)
        print(f"\nPERIOD MODE: {hits}/{len(periods)} hand-scored periods overlap a detected "
              f"anticipatory bout  (recall {hits/max(len(periods),1):.1%})")
        if misses:
            print("  missed periods at:",
                  ", ".join(f"{periods[i][0]:.2f}s" for i in misses[:12]),
                  "..." if len(misses) > 12 else "")
        return 0

    # ---------------- per-lick mode ----------------
    offset = 0.0
    if args.offset_scan:
        offs = np.arange(-args.offset_range, args.offset_range + 1e-9, args.offset_step)
        best_n, best_o = -1, 0.0
        for o in offs:
            p, _, _ = greedy_match(gt + o, det, args.match_tolerance)
            if len(p) > best_n:
                best_n, best_o = len(p), float(o)
        offset = best_o
        print(f"\noffset scan  : best offset {offset:+.3f}s  ({best_n} matches; "
              f"{len(greedy_match(gt, det, args.match_tolerance)[0])} at zero offset)")

    pairs, gt_un, det_un = greedy_match(gt + offset, det, args.match_tolerance)
    tp, fn, fp = len(pairs), len(gt_un), len(det_un)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    print(f"\nMATCHING (tolerance +/-{args.match_tolerance}s"
          + (f", offset {offset:+.3f}s" if offset else "") + ")")
    print(f"  matched (TP)                : {tp}")
    print(f"  hand-scored but missed (FN) : {fn}")
    print(f"  detected but not scored (FP): {fp}")
    print(f"  precision {prec:.3f}   recall {rec:.3f}   F1 {f1:.3f}")
    if pairs:
        d = np.array([p[2] for p in pairs])
        print(f"  time offset of matches: median {np.median(d):+.3f}s  "
              f"IQR [{np.percentile(d,25):+.3f}, {np.percentile(d,75):+.3f}]s")
        if abs(np.median(d)) > args.match_tolerance / 2:
            print("    NOTE: large median offset suggests a systematic clock shift "
                  "(try --offset-scan), not a detection failure.")

    # ---------------- why did we miss them? ----------------
    rows = []
    if fn and bouts is not None and len(bouts):
        bs = bouts["start_s"].to_numpy(float)
        be = bouts["end_s"].to_numpy(float)
        bc = bouts["classification"].to_numpy()
        tally = {}
        for gi in gt_un:
            t = gt[gi] + offset
            inside = np.where((t >= bs - args.match_tolerance) &
                              (t <= be + args.match_tolerance))[0]
            where = bc[inside[0]] if len(inside) else "no detected lick"
            tally[where] = tally.get(where, 0) + 1
            rows.append(dict(gt_time_s=round(gt[gi], 3), matched=False, fell_inside=where))
        print("\n  WHY the missed hand scores were missed:")
        for k2, v in sorted(tally.items(), key=lambda x: -x[1]):
            label = {"consummatory": "detected but classified CONSUMMATORY (classification error)",
                     "other": "detected but classified OTHER (not tied to an upcoming reward)",
                     "anticipatory": "inside an anticipatory bout but outside tolerance",
                     "no detected lick": "no tongue detected there at all (detector miss)"}.get(k2, k2)
            print(f"    {v:5d}  {label}")

    for gi, di, dt in pairs:
        rows.append(dict(gt_time_s=round(gt[gi], 3), matched=True,
                         detected_time_s=round(det[di], 3), offset_s=round(dt, 3)))
    if args.out and rows:
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"\nwrote per-hand-score results to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
