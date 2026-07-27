#!/usr/bin/env python3
"""
review_bouts.py
===============

Build an annotated **contact sheet** to visually verify the lick classifications against the
actual video. For each bout it grabs the peak-tongue frame, draws the tracked points
(tongue / spout / nose) plus the learned retracted (R) and extended (E) spout regions, crops
a zoom around the mouth, captions it with bout id / class / timestamp, and tiles everything
into one or more PNG pages.

How to read a cell
------------------
* RED dot   = tracked tongue (is there really an extended tongue there? -> precision check)
* CYAN dot  = tracked spout tip
* hollow R  = retracted spout region, hollow E = extended spout region
  -> for an ANTICIPATORY bout the cyan dot should sit at R (spout away from mouth);
     for a CONSUMMATORY bout it should sit at E (spout up at the mouth).

Usage
-----
    python review_bouts.py VIDEO.mp4 --bouts <stem>_lick_bouts.csv \
        --sleap "<the SLEAP analysis>.csv" [--class anticipatory] [--out review]

Requires: opencv-python, pandas, numpy (and detect_anticipatory_licks.py alongside this file).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import cv2

from detect_anticipatory_licks import load_sleap_csv, learn_spout_regions, DEFAULTS, _dist


# BGR colors
C_TONGUE = (0, 0, 255)      # red
C_SPOUT = (255, 255, 0)     # cyan
C_NOSE = (0, 255, 255)      # yellow
C_R = (255, 255, 255)       # white  (retracted region anchor)
C_E = (0, 255, 0)           # green  (extended region anchor)
CLS_COLOR = {"anticipatory": (0, 0, 255), "consummatory": (0, 200, 0),
             "other": (0, 200, 255)}   # amber: retracted licking not tied to a reward


def _xy(row, node):
    x, y = row.get(f"{node}.x"), row.get(f"{node}.y")
    if x is None or y is None or not np.isfinite(x) or not np.isfinite(y):
        return None
    return int(round(x)), int(round(y))


def peak_frame_of_bout(sleap_by_frame, start, end):
    """Frame in [start, end] with the highest tongue.score (falls back to start)."""
    sub = sleap_by_frame.loc[
        (sleap_by_frame.index >= start) & (sleap_by_frame.index <= end)]
    if len(sub) == 0:
        return start, 0.0
    sc = sub["tongue.score"].fillna(0.0)
    f = int(sc.idxmax())
    return f, float(sc.loc[f]) if hasattr(sc.loc[f], "__float__") else float(sc.max())


def render_cell(frame_img, srow, reg, crop_center, crop_half, thumb, caption, cls):
    """Draw points + region anchors on a copy of the frame, crop/resize, add a caption bar."""
    img = frame_img.copy()
    # points
    for node, col, r in [("nose", C_NOSE, 3), ("spout_tip", C_SPOUT, 6), ("tongue", C_TONGUE, 6)]:
        p = _xy(srow, node)
        if p:
            cv2.circle(img, p, r, col, -1)
            cv2.circle(img, p, r + 1, (0, 0, 0), 1)
    # region anchors
    rc = (int(reg.retracted[0]), int(reg.retracted[1]))
    cv2.circle(img, rc, 7, C_R, 2)
    cv2.putText(img, "R", (rc[0] + 8, rc[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_R, 1, cv2.LINE_AA)
    if reg.extended is not None:
        ec = (int(reg.extended[0]), int(reg.extended[1]))
        cv2.circle(img, ec, 7, C_E, 2)
        cv2.putText(img, "E", (ec[0] + 8, ec[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_E, 1, cv2.LINE_AA)

    # crop a zoom around the mouth/spout area, clipped to frame bounds
    h, w = img.shape[:2]
    cx, cy = crop_center
    x0 = max(0, int(cx - crop_half)); x1 = min(w, int(cx + crop_half))
    y0 = max(0, int(cy - crop_half)); y1 = min(h, int(cy + crop_half))
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        crop = img
    cell = cv2.resize(crop, (thumb, thumb), interpolation=cv2.INTER_AREA)

    # caption bar
    bar = np.zeros((26, thumb, 3), dtype=np.uint8)
    cv2.putText(bar, caption, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                CLS_COLOR.get(cls, (255, 255, 255)), 1, cv2.LINE_AA)
    return np.vstack([cell, bar])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Annotated contact sheet to verify lick bouts.")
    ap.add_argument("video", help="the Cam video (mp4)")
    ap.add_argument("--bouts", required=True, help="<stem>_lick_bouts.csv from the detector")
    ap.add_argument("--sleap", required=True, help="the SLEAP analysis CSV (for keypoints)")
    ap.add_argument("--class", dest="cls", default="anticipatory",
                    choices=["anticipatory", "consummatory", "other", "all"],
                    help="which bouts to render (default: anticipatory). 'other' = retracted "
                         "licking not leading into a reward, useful for checking the split")
    ap.add_argument("--out", default="review", help="output PNG base name")
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--rows-per-page", type=int, default=6)
    ap.add_argument("--thumb", type=int, default=220, help="thumbnail size in px")
    ap.add_argument("--max", type=int, default=None, help="cap number of bouts rendered")
    ap.add_argument("--every", type=int, default=1, help="render every Nth bout (sampling)")
    args = ap.parse_args(argv)

    if not os.path.exists(args.video):
        print(f"video not found: {args.video}")
        return 1

    sleap = load_sleap_csv(args.sleap)
    sleap_by_frame = sleap.set_index("frame_idx")
    cfg = dict(DEFAULTS)
    reg = learn_spout_regions(sleap, cfg)
    print(f"spout regions: retracted=({reg.retracted[0]:.0f},{reg.retracted[1]:.0f}) "
          f"extended={'none' if reg.extended is None else f'({reg.extended[0]:.0f},{reg.extended[1]:.0f})'}")

    # crop geometry: a zoom around the mouth, scaled per-video so it includes the spout's
    # retracted and extended positions (works at any camera angle/zoom)
    nose_med = np.array([np.nanmedian(sleap["nose.x"]), np.nanmedian(sleap["nose.y"])])
    anchor = reg.extended if reg.extended is not None else reg.retracted
    scale = max(_dist(nose_med[0], nose_med[1], reg.retracted[0], reg.retracted[1]),
                _dist(nose_med[0], nose_med[1], anchor[0], anchor[1]), 40.0)
    crop_center = ((nose_med + reg.retracted) / 2.0)
    crop_half = 1.6 * scale

    bouts = pd.read_csv(args.bouts)
    if args.cls != "all":
        bouts = bouts[bouts["classification"] == args.cls]
    bouts = bouts.iloc[::args.every]
    if args.max:
        bouts = bouts.head(args.max)
    print(f"rendering {len(bouts)} '{args.cls}' bouts")
    if len(bouts) == 0:
        print("nothing to render.")
        return 0

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"could not open video: {args.video}")
        return 1

    cells = []
    for _, b in bouts.iterrows():
        pf, sc = peak_frame_of_bout(sleap_by_frame, int(b["start_frame"]), int(b["end_frame"]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, pf)
        ok, img = cap.read()
        if not ok or img is None:
            img = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480,
                            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640, 3), np.uint8)
        srow = sleap_by_frame.loc[pf] if pf in sleap_by_frame.index else {}
        tid = int(b["trial_id"]) if "trial_id" in b and pd.notna(b["trial_id"]) else 0
        cap_txt = (f"#{int(b['bout_id'])} T{tid} {b['classification'][:5]} "
                   f"t={b.get('start_s', pf):.1f}s s={sc:.2f}")
        cells.append(render_cell(img, srow, reg, crop_center, crop_half,
                                 args.thumb, cap_txt, b["classification"]))
    cap.release()

    def legend_bar(width):
        bar = np.full((28, width, 3), 25, np.uint8)
        x = 8
        for txt, col in [("tongue", C_TONGUE), ("spout", C_SPOUT), ("nose", C_NOSE),
                         ("R=retracted", C_R), ("E=extended", C_E)]:
            cv2.circle(bar, (x + 6, 14), 6, col, -1)
            cv2.putText(bar, txt, (x + 16, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (230, 230, 230), 1, cv2.LINE_AA)
            x += 20 + len(txt) * 11
        cv2.putText(bar, "caption: #bout  Ttrial  class  t=start  s=tongue_score",
                    (max(x, width - 420), 19), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (180, 180, 180), 1, cv2.LINE_AA)
        return bar

    # tile into pages
    per_page = args.cols * args.rows_per_page
    n_pages = int(np.ceil(len(cells) / per_page))
    pad = 4
    for pi in range(n_pages):
        page_cells = cells[pi * per_page:(pi + 1) * per_page]
        ch, cw = page_cells[0].shape[:2]
        rows = []
        for r in range(0, len(page_cells), args.cols):
            row_cells = page_cells[r:r + args.cols]
            while len(row_cells) < args.cols:
                row_cells.append(np.zeros_like(page_cells[0]))
            row = np.hstack([cv2.copyMakeBorder(c, pad, pad, pad, pad,
                                                cv2.BORDER_CONSTANT, value=(40, 40, 40))
                             for c in row_cells])
            rows.append(row)
        body = np.vstack(rows)
        sheet = np.vstack([legend_bar(body.shape[1]), body])
        out = f"{args.out}_{args.cls}_page{pi + 1}.png" if n_pages > 1 else f"{args.out}_{args.cls}.png"
        cv2.imwrite(out, sheet)
        print(f"wrote {out}  ({len(page_cells)} bouts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
