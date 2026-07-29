#!/usr/bin/env python3
"""
detect_anticipatory_licks.py
============================

Detect *anticipatory* licks (the licking run that leads up to a reward) vs. *consummatory*
licks (licking while the spout is extended) from SLEAP tongue/spout tracking on head-fixed
oHrbets videos.

Rig logic that makes this work
------------------------------
The lick spout physically extends (~1 inch, toward the mouth) only after the mouse has
earned a reward, and is retracted otherwise -- so each spout extension marks a reward trial.
Licks are classified by where they fall in that trial structure:

    [mouse starts licking .... spout EXTENDS)  -> ANTICIPATORY licks
    [spout EXTENDS .......... spout RETRACTS)  -> CONSUMMATORY licks
    any other licking (spout retracted, not    -> 'other': kept in the full bout table but
     leading into an extension)                   EXCLUDED from the anticipatory outputs

Anticipatory licking is therefore specifically the run that *leads into* a reward, not merely
any licking that happens while the spout is away.

Cross-camera generalization
---------------------------
Different videos use different camera angles/zoom, so NO pixel coordinate or pixel
threshold is hardcoded. Each video is calibrated from its own data:
  * the two spout "areas of interest" (retracted vs extended) are learned by clustering
    that video's spout_tip point cloud (robust median + secondary-density-peak recipe);
  * per-frame spout state = nearest learned region by Euclidean distance (angle-invariant);
  * the tongue false-positive ROI is learned from that video's tongue cloud relative to
    the nose.
The only constant that legitimately transfers across cameras is the model-confidence
threshold (tongue.score), which is a property of the model, not the geometry.

Outputs
-------
`<stem>_anticipatory_licks.csv`  PRIMARY -- one row per *individual* anticipatory tongue
                                 protrusion (comparable to per-lick hand scoring)
`<stem>_anticipatory_bouts.csv`  one row per anticipatory licking period
`<stem>_lick_bouts.csv`          every bout, all three classes, with trial_id
`<stem>_trials.csv`              one row per spout extension (the trial structure)

The rig touch-sensor timestamps are used (optionally) only to *verify* the classification.

Only input required
-------------------
The SLEAP analysis CSV. Nothing else -- not the video, not the .slp, not any rig log. The job
here is narrow on purpose: take the tongue and spout the model reports, filter out what is not
a real lick, and sort what remains by where it falls relative to reward delivery, so the model
can be judged on how well it picks up anticipatory licking. Only pandas and numpy are needed.

Usage
-----
    python detect_anticipatory_licks.py INPUT [options]

INPUT may be a single SLEAP analysis CSV or a folder (every analysis CSV inside it is
processed, with a combined cross-session summary written too). Files this script has already
written are skipped, so re-running over the same folder is safe.

Run `python detect_anticipatory_licks.py --help` for all options.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# Defaults (all overridable on the command line)
# --------------------------------------------------------------------------------------
DEFAULTS = dict(
    fps=60.0,                 # confirmed by PI for the Cam1 videos
    spout_score_min=0.5,      # min spout_tip.score to trust a spout point for state
    spout_calib_score_min=0.6,  # min score for points used to LEARN the regions
    tongue_score_min=0.30,    # model-confidence gate for a real tongue (camera-independent)
    tongue_roi_k=8.0,         # tongue ROI: reject if nose-relative dist > median + k*MAD
    min_protrusion_frames=2,  # a tongue run must persist >= this many frames (kill blips)
    bout_gap_frames=20,       # merge protrusions <= this many frames apart into one bout
    min_extended_run=5,       # an extension must persist >= this many frames (OPEN)
    extension_merge_gap=30,   # bridge extended runs <= this many frames apart (CLOSE)
    lick_episode_gap=120,     # licks <= this many frames apart are one licking episode (2s)
    anticipatory_max_gap=120, # max gap from last lick to the extension onset to count as lead-in
    post_reward_s=2.0,        # licking within this long after a retraction is reward clean-up,
                              # not anticipation. Set to 0 to count it as anticipatory.
    min_extended_frac=0.002,  # an extended region must hold >= this frac of spout points
    min_extension_s=None,     # optional: drop extensions shorter than this many seconds
    max_extension_s=None,     # optional: drop extensions longer than this many seconds
    trim_start_s=None,        # optional: ignore extensions in the first N s (auto if None)
    trim_end_s=None,          # optional: ignore extensions in the last N s (auto if None)
    outlier_lo=0.5,           # flag trials whose duration < this * median duration
    outlier_hi=2.0,           # flag trials whose duration > this * median duration
)

NODES = ["nose", "tongue", "spout_tip", "left_paw", "right_paw"]

# Everything this script writes. The names keep the source stem, which itself contains
# "analysis", so they match the same glob the inputs do and would otherwise be re-ingested on a
# second run over the same folder.
OUTPUT_SUFFIXES = ("_anticipatory_licks.csv", "_anticipatory_bouts.csv", "_lick_bouts.csv",
                   "_trials.csv", "_frames_annotated.csv", "combined_session_summary.csv")


def is_output_file(path: str) -> bool:
    return os.path.basename(path).endswith(OUTPUT_SUFFIXES)


def find_input_csvs(folder: str):
    """SLEAP analysis CSVs in `folder`, skipping this script's own output."""
    hits = sorted(glob.glob(os.path.join(folder, "*analysis*.csv")))
    if not hits:
        hits = sorted(glob.glob(os.path.join(folder, "*.csv")))
    return [f for f in hits if not is_output_file(f)]


# --------------------------------------------------------------------------------------
# Small geometry / stats helpers
# --------------------------------------------------------------------------------------
def _dist(ax, ay, bx, by):
    return np.hypot(ax - bx, ay - by)


def _mad(x):
    """Median absolute deviation, scaled to be a robust std estimate."""
    x = np.asarray(x, float)
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))


# --------------------------------------------------------------------------------------
# Step 1 -- load
# --------------------------------------------------------------------------------------
def load_sleap_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "frame_idx" not in df.columns:
        raise ValueError(f"{path}: not a SLEAP analysis CSV (no 'frame_idx' column)")
    missing = [f"{n}.x" for n in NODES if f"{n}.x" not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing expected node columns: {missing}")
    df = df.sort_values("frame_idx").reset_index(drop=True)

    # Every temporal parameter (episode gaps, extension merge gap, min run, durations) is
    # expressed in FRAMES but applied to row positions, so a CSV that omits frames would
    # silently mis-measure time. SLEAP can drop frames with no detected instance, so make the
    # frame axis contiguous up front; the inserted rows are all-NaN and fall out naturally.
    fi = df["frame_idx"].to_numpy()
    if len(fi):
        full = np.arange(fi.min(), fi.max() + 1)
        if len(fi) != len(full) or not np.array_equal(fi, full):
            print(f"  NOTE: {len(full) - len(fi)} frame(s) absent from the CSV; reindexing "
                  f"onto a contiguous frame axis so frame-based gaps stay correct")
            df = (df.drop_duplicates("frame_idx", keep="first")
                    .set_index("frame_idx").reindex(full)
                    .rename_axis("frame_idx").reset_index())
    return df


# --------------------------------------------------------------------------------------
# Step 2 -- learn the two spout regions ("areas of interest") by clustering
# --------------------------------------------------------------------------------------
@dataclass
class SpoutRegions:
    retracted: np.ndarray                 # (x, y)
    extended: np.ndarray | None           # (x, y) or None if not found
    separation: float = 0.0               # px between centroids
    n_calib: int = 0
    ext_support: int = 0                  # frames sitting at the extended position
    radius: float = 0.0                   # how close counts as "at" a position
    note: str = ""


def _blob_edge(disp: np.ndarray) -> float:
    """Radius at which the dominant (retracted) cloud has thinned out.

    Derived from the data, so it scales with the camera's zoom instead of assuming pixels.
    """
    inner = disp[disp <= np.percentile(disp, 99.9)]
    if len(inner) < 50:
        return float(np.percentile(disp, 99)) if len(disp) else 0.0
    nb = 60
    hist, edges = np.histogram(inner, bins=nb)
    centers = 0.5 * (edges[:-1] + edges[1:])
    kern = np.array([1, 2, 3, 2, 1], float)
    sm = np.convolve(hist, kern / kern.sum(), mode="same")
    peak = int(np.argmax(sm))
    thr = 0.02 * sm[peak]
    i = peak
    while i < nb and sm[i] >= thr:
        i += 1
    return float(centers[min(i, nb - 1)])


def _position_clusters(P: np.ndarray, radius: float, max_k: int = 8):
    """Greedy density clustering of 2-D points. Returns [(centroid, n_points), ...].

    Repeatedly takes the densest cell of a coarse histogram, absorbs everything within
    `radius` of it, and removes those points.
    """
    out = []
    rest = P
    while len(rest) >= 10 and len(out) < max_k:
        bins = max(4, int(np.ptp(rest, axis=0).max() / max(radius, 1e-6)) + 1)
        H, xe, ye = np.histogram2d(rest[:, 0], rest[:, 1], bins=bins)
        bx, by = np.unravel_index(int(np.argmax(H)), H.shape)
        seed = np.array([(xe[bx] + xe[bx + 1]) / 2, (ye[by] + ye[by + 1]) / 2])
        near = _dist(rest[:, 0], rest[:, 1], seed[0], seed[1]) <= radius
        if not near.any():
            break
        c = np.median(rest[near], axis=0)                     # refine on the members
        near = _dist(rest[:, 0], rest[:, 1], c[0], c[1]) <= radius
        out.append((c, int(near.sum())))
        rest = rest[~near]
    return out


def _runs_at(sx, sy, sc, centre, radius, cfg):
    """Contiguous runs where the spout sits at `centre`, after closing detector flicker."""
    at = (_dist(sx, sy, centre[0], centre[1]) <= radius) & \
         (sc >= cfg["spout_score_min"]) & np.isfinite(sx)
    merged = []
    for s, e in _mask_runs(at):
        if merged and s - merged[-1][1] <= cfg["extension_merge_gap"]:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return [(s, e) for s, e in merged if (e - s) >= cfg["min_extended_run"]]


def learn_spout_regions(df: pd.DataFrame, cfg: dict,
                        ret_override=None, ext_override=None) -> SpoutRegions:
    """Find the retracted position and the reward-delivery position for THIS video.

    The retracted position is easy -- it dominates the recording, so a component-wise median
    of the high-confidence spout points nails it and shrugs off outliers.

    The reward position is the subtle one. Picking the biggest cluster away from retracted is
    wrong: raw recordings open with a setup period where the spout is parked somewhere else or
    swung around by hand, and that can outweigh the real trials by frame count many times over.
    What actually identifies reward delivery is that the rig drives the spout to the *same*
    place for the *same* length of time, over and over. So each candidate position is scored by
    how many repeat visits it gets at a consistent duration, and the winner takes it.
    """
    hi = df[df["spout_tip.score"] > cfg["spout_calib_score_min"]].dropna(
        subset=["spout_tip.x", "spout_tip.y"])
    X = hi[["spout_tip.x", "spout_tip.y"]].to_numpy(float)

    if ret_override is not None:
        ret = np.asarray(ret_override, float)
    elif len(X) == 0:
        raise ValueError("no high-confidence spout points to calibrate regions")
    else:
        ret = np.median(X, axis=0)

    disp = _dist(X[:, 0], X[:, 1], ret[0], ret[1])
    edge = _blob_edge(disp)
    radius = max(edge, 1.0)

    if ext_override is not None:
        ext = np.asarray(ext_override, float)
        sep = float(_dist(ret[0], ret[1], ext[0], ext[1]))
        return SpoutRegions(ret, ext, sep, len(X), 0, radius, "extended position set by override")

    if len(X) < 50:
        return SpoutRegions(ret, None, 0.0, len(X), 0, radius, "too few spout points to calibrate")

    far = X[disp > edge]
    n_min = max(20, cfg["min_extended_frac"] * len(X))
    if len(far) < n_min:
        return SpoutRegions(ret, None, 0.0, len(X), len(far), radius,
                            "spout never leaves its resting position")

    sx = df["spout_tip.x"].to_numpy(float)
    sy = df["spout_tip.y"].to_numpy(float)
    sc = df["spout_tip.score"].to_numpy(float)

    best, best_key, fallback, fb_key, notes = None, None, None, None, []
    for c, npts in _position_clusters(far, radius):
        sep_c = _dist(c[0], c[1], ret[0], ret[1])
        # A real extension throws the spout clear of its resting jitter. Anything hugging the
        # rest blob is the spout in transit, not a place it is driven to and held.
        if sep_c <= 2 * edge:
            continue
        runs = _runs_at(sx, sy, sc, c, radius, cfg)
        if not runs:
            continue
        d = np.array([e - s for s, e in runs], float)
        med = float(np.median(d))
        n_rep = int(np.sum(np.abs(d - med) <= 0.25 * med))
        agree = n_rep / len(runs)
        notes.append(f"({c[0]:.0f},{c[1]:.0f}) {len(runs)} visits, {n_rep} at ~{med/cfg['fps']:.2f}s")
        if fb_key is None or npts > fb_key:
            fallback, fb_key = c, npts
        # The rig drives the spout to the reward position for a fixed time, so nearly every
        # visit should share one duration. Points along the travel path collect many visits of
        # scattered length, which is what this rejects.
        if n_rep >= 2 and agree >= 0.5:
            key = (n_rep, agree, sep_c)
            if best_key is None or key > best_key:
                best, best_key = c, key

    if best is None and fallback is not None:
        best = fallback
        notes.append("no position had repeatable visit lengths; fell back to the most occupied")
    if best is None:
        return SpoutRegions(ret, None, 0.0, len(X), 0, radius,
                            "no repeated spout position found away from rest")

    sep = float(_dist(ret[0], ret[1], best[0], best[1]))
    support = int(np.sum(_dist(far[:, 0], far[:, 1], best[0], best[1]) <= radius))
    note = f"rest-blob edge {edge:.0f}px; chose " + notes[0] if len(notes) == 1 else \
           f"rest-blob edge {edge:.0f}px; candidates " + " | ".join(notes)
    return SpoutRegions(ret, best, sep, len(X), support, radius, note)


# --------------------------------------------------------------------------------------
# Step 3 -- extension events ("trials") and the per-frame spout state derived from them
# --------------------------------------------------------------------------------------
def _mask_runs(mask: np.ndarray):
    """Contiguous [start, end) runs of True."""
    d = np.diff(mask.astype(np.int8))
    s = list(np.where(d == 1)[0] + 1)
    e = list(np.where(d == -1)[0] + 1)
    if len(mask) and mask[0]:
        s = [0] + s
    if len(mask) and mask[-1]:
        e = e + [len(mask)]
    return list(zip(s, e))


def find_extension_events(df: pd.DataFrame, reg: SpoutRegions, cfg: dict):
    """Find the spout-extension events (reward trials) as [onset, offset) index intervals.

    The spout detector flickers, so a single physical extension often appears as several
    short runs separated by 1-5 frame gaps, while genuine inter-trial gaps are seconds long.
    We therefore apply a morphological CLOSE then OPEN, *in that order*:

      1. raw extended mask = nearest region is 'extended' and the spout point is trustworthy;
      2. CLOSE  -- merge runs separated by <= `extension_merge_gap` (bridges flicker);
      3. OPEN   -- drop runs shorter than `min_extended_run` (kills spurious blips).

    Doing OPEN before CLOSE (the previous behaviour) destroys fragmented real extensions and
    under-counts trials, which matters because every anticipatory window is anchored to an
    extension onset.
    """
    if reg.extended is None:
        return []

    sx = df["spout_tip.x"].to_numpy(float)
    sy = df["spout_tip.y"].to_numpy(float)
    sc = df["spout_tip.score"].to_numpy(float)
    # An extension is the spout sitting AT the learned reward position, not merely "closer to
    # extended than to retracted". Recordings contain other spout placements (setup, hand
    # adjustment); a nearest-of-two rule would sweep those in as rewards.
    trials = _runs_at(sx, sy, sc, reg.extended, reg.radius, cfg)   # CLOSE then OPEN

    # optional duration gate -- the rig extends the spout for a fixed time (~3s on the sample),
    # so wildly off-duration "extensions" are artifacts (e.g. the spout parked out before the
    # session starts, or a 2-frame blip). Off by default so nothing is dropped silently.
    fps = cfg["fps"]
    lo = cfg.get("min_extension_s")
    hi = cfg.get("max_extension_s")
    if lo is not None:
        trials = [(s, e) for s, e in trials if (e - s) / fps >= lo]
    if hi is not None:
        trials = [(s, e) for s, e in trials if (e - s) / fps <= hi]
    return trials


def assess_trials(trials, cfg, n_frames):
    """Decide which detected extensions are real reward trials.

    These are raw, uncut recordings: at the very start the mouse is still being put on the rig
    (the spout can sit parked out) and near the end it is being taken off, so extensions at the
    edges are unreliable and must not be given the same weight as mid-session ones.

    Two independent checks, both data-driven (no hardcoded seconds):
      * duration  -- the rig extends the spout for a fixed time, so a wildly off-median
        duration is an artifact, not a reward;
      * position  -- the active session is taken to span the well-formed trials, so anything
        outside that span is edge-of-recording noise.

    `--trim-start-s` / `--trim-end-s` override the position check explicitly.
    Returns a list of per-trial dicts (same order as `trials`).
    """
    if not trials:
        return []
    fps = cfg["fps"]
    dur = np.array([e - s for s, e in trials], float)
    med = float(np.median(dur))
    outlier = (dur < cfg["outlier_lo"] * med) | (dur > cfg["outlier_hi"] * med)

    ok = ~outlier
    if ok.any():
        lo_default = trials[int(np.argmax(ok))][0]                      # first well-formed trial
        hi_default = trials[len(ok) - 1 - int(np.argmax(ok[::-1]))][1]  # last well-formed trial
    else:
        lo_default, hi_default = 0, n_frames

    ts, te = cfg.get("trim_start_s"), cfg.get("trim_end_s")
    lo = ts * fps if ts is not None else lo_default
    hi = n_frames - te * fps if te is not None else hi_default

    out = []
    for i, (s, e) in enumerate(trials):
        outside = bool(e <= lo or s >= hi)
        out.append(dict(
            duration_s=round(dur[i] / fps, 3),
            duration_outlier=bool(outlier[i]),
            outside_session=outside,
            accepted=bool(not outlier[i] and not outside),
        ))
    return out


def session_bounds(df: pd.DataFrame, reg: SpoutRegions, trials, accepted, cfg, n: int):
    """Frame range over which licking can be scored, as [lo, hi).

    Raw recordings open with the animal being mounted and the spout being placed or primed by
    hand, and close with it being taken off. During setup the spout is often held at the mouth
    at some position that is not the reward position at all, so the mouse is licking a spout
    that is right there. Counting that as anticipation would be wrong.

    Any run with the spout away from rest that is not one of the accepted reward trials is
    treated as such an off-protocol placement. The session therefore starts after the last one
    of those preceding the first reward, and runs to the last reward plus the post-reward
    window.
    """
    acc = [t for t, ok in zip(trials, accepted) if ok]
    if not acc:
        return 0, n
    first_s, last_e = acc[0][0], acc[-1][1]
    pr = int(round(cfg["post_reward_s"] * cfg["fps"]))

    lo = 0
    if reg.extended is not None:
        sx = df["spout_tip.x"].to_numpy(float)
        sy = df["spout_tip.y"].to_numpy(float)
        sc = df["spout_tip.score"].to_numpy(float)
        away = (_dist(sx, sy, reg.retracted[0], reg.retracted[1]) > reg.radius) & \
               (sc >= cfg["spout_score_min"]) & np.isfinite(sx)
        merged = []
        for s, e in _mask_runs(away):
            if merged and s - merged[-1][1] <= cfg["extension_merge_gap"]:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))
        acc_starts = {s for s, _ in acc}
        for s, e in merged:
            if (e - s) < cfg["min_extended_run"] or e > first_s:
                continue
            if s in acc_starts:            # a genuine reward trial, not a placement
                continue
            lo = max(lo, e)

    hi = min(n, last_e + max(pr, 1))
    if cfg.get("trim_start_s") is not None:
        lo = max(lo, int(round(cfg["trim_start_s"] * cfg["fps"])))
    if cfg.get("trim_end_s") is not None:
        hi = min(hi, n - int(round(cfg["trim_end_s"] * cfg["fps"])))
    return lo, max(hi, lo)


def state_from_trials(n: int, trials, index=None) -> pd.Series:
    """Per-frame spout state derived from the final trial intervals, so the state and the
    trial list can never disagree."""
    st = np.array(["retracted"] * n, dtype=object)
    for s, e in trials:
        st[s:e] = "extended"
    return pd.Series(st, index=index if index is not None else np.arange(n))


# --------------------------------------------------------------------------------------
# Step 4 -- clean tongue detections (false-positive filter)
# --------------------------------------------------------------------------------------
def clean_tongue(df: pd.DataFrame, cfg: dict):
    """Return (valid_mask, info). A frame's tongue is 'valid' if it clears the score gate,
    sits inside a learned nose-relative ROI, and belongs to a run >= min_protrusion_frames.
    """
    tx = df["tongue.x"].to_numpy(float)
    ty = df["tongue.y"].to_numpy(float)
    ts = df["tongue.score"].to_numpy(float)
    nx = df["nose.x"].to_numpy(float)
    ny = df["nose.y"].to_numpy(float)

    # The nose can be missing on many frames (~17% here). Without a fallback the ROI silently
    # cannot gate those frames, so low-confidence junk detections slip through un-validated --
    # exactly the frames most likely to be false positives. This is a head-FIXED prep, so the
    # nose is essentially stationary (std ~2-11px), making the session median a sound anchor.
    n_nose_missing = int((~np.isfinite(nx)).sum())
    if np.isfinite(nx).any():
        nx = np.where(np.isfinite(nx), nx, np.nanmedian(nx))
        ny = np.where(np.isfinite(ny), ny, np.nanmedian(ny))

    has_tongue = np.isfinite(tx) & np.isfinite(ty) & (ts >= cfg["tongue_score_min"])

    # learn the ROI from score-passing tongue points relative to the nose (scale/angle aware)
    d_nose = _dist(tx, ty, nx, ny)
    learn = has_tongue & np.isfinite(d_nose)
    if learn.sum() >= 20:
        med = np.median(d_nose[learn])
        mad = _mad(d_nose[learn]) or 1.0
        roi_cut = med + cfg["tongue_roi_k"] * mad
    else:
        roi_cut = np.inf  # too few points to learn an ROI; don't reject spatially
    in_roi = ~np.isfinite(d_nose) | (d_nose <= roi_cut)  # missing nose -> don't reject

    valid = has_tongue & in_roi
    # require runs of >= min_protrusion_frames consecutive valid frames
    valid = _min_run_mask(valid, cfg["min_protrusion_frames"])

    info = dict(n_score_pass=int(has_tongue.sum()),
                n_roi_rejected=int((has_tongue & ~in_roi).sum()),
                roi_cut_px=float(roi_cut) if np.isfinite(roi_cut) else None,
                n_nose_missing=n_nose_missing,
                n_valid=int(valid.sum()))
    return valid, info


def _min_run_mask(mask: np.ndarray, min_len: int) -> np.ndarray:
    if min_len <= 1:
        return mask
    out = mask.copy()
    n = len(out)
    i = 0
    while i < n:
        if out[i]:
            j = i
            while j < n and out[j]:
                j += 1
            if (j - i) < min_len:
                out[i:j] = False
            i = j
        else:
            i += 1
    return out


# --------------------------------------------------------------------------------------
# Step 5 & 6 -- group protrusions into bouts and classify
# --------------------------------------------------------------------------------------
@dataclass
class Protrusion:
    """One individual tongue-out event (a run of consecutive clean tongue frames)."""
    protrusion_id: int
    trial_id: int
    bout_id: int
    label: str
    onset_frame: int
    peak_frame: int
    offset_frame: int
    n_frames: int
    max_score: float
    leads_into_reward: bool = False   # part of the run that builds straight into the extension
    isolated: bool = False            # the only lick in its bout
    s_to_extension: float = float("nan")   # seconds until this trial's extension


@dataclass
class Bout:
    bout_id: int
    trial_id: int
    classification: str
    start_frame: int
    end_frame: int
    n_licks: int
    n_tongue_frames: int
    leads_into_reward: bool = False
    mean_spout_disp: float = 0.0
    mean_tongue_score: float = 0.0
    tongue_x: float = 0.0
    tongue_y: float = 0.0


def derive_anticipatory_windows(valid: np.ndarray, trials, cfg: dict, accepted=None):
    """For each trial, find the licking episode that leads into the spout extension.

    Returns a list aligned with `trials`: either `(start_idx, onset_idx)` or None when the
    mouse was not licking into that extension (or the trial is a rejected edge/duration
    artifact, which never gets an anticipatory window).

    The candidate licks are clipped to the *previous* trial's offset, so licking that belongs
    to the previous trial's consummatory / post-reward period can never be absorbed into this
    trial's anticipatory window. Rejected trials still clip (the spout really was out then),
    they just do not earn a window of their own.
    """
    lick = np.where(valid)[0]
    windows = []
    prev_off = 0
    for ti, (s, e) in enumerate(trials):
        win = None
        if accepted is None or accepted[ti]:
            cand = lick[(lick < s) & (lick >= prev_off)]
            if len(cand) and (s - cand[-1]) <= cfg["anticipatory_max_gap"]:
                i = len(cand) - 1
                while i > 0 and (cand[i] - cand[i - 1]) <= cfg["lick_episode_gap"]:
                    i -= 1
                win = (int(cand[i]), int(s))
        windows.append(win)
        prev_off = e
    return windows


def label_licks(n: int, trials, windows, cfg, accepted=None, bounds=None):
    """Per-frame label, trial id, and a lead-in flag.

    A lick is ANTICIPATORY whenever the spout is retracted during the active session and the
    mouse is not just finishing a reward. It does NOT have to be part of an established run of
    licking -- the mouse often makes a single isolated lick between trials, and that is still a
    lick made with no reward available, so it counts.

    The lead-in runs found by `derive_anticipatory_windows` are therefore no longer used to
    *include* licks. They only set the `leads_into_reward` flag, so a run that builds straight
    into an extension can still be told apart from a lone spontaneous lick downstream.

      consummatory  inside an accepted extension
      post_reward   within `post_reward_s` of an extension ending (finishing the reward,
                    so not anticipation). Set --post-reward-s 0 to fold these in.
      anticipatory  spout retracted, inside the active session, everything else
      other         outside the active session, or inside a rejected extension

    Returns (labels, trial_id, leads_into_reward).
    """
    labels = np.array(["other"] * n, dtype=object)
    trial_id = np.zeros(n, dtype=int)
    lead = np.zeros(n, dtype=bool)
    if accepted is None:
        accepted = [True] * len(trials)

    acc = [(i + 1, s, e) for i, (s, e) in enumerate(trials) if accepted[i]]
    if not acc:
        return labels, trial_id, lead

    pr = int(round(cfg["post_reward_s"] * cfg["fps"]))
    last_e = acc[-1][2]
    lo, hi = bounds if bounds is not None else (0, n)

    # Everything inside the session with the spout retracted starts out anticipatory, and is
    # attributed to the next extension it precedes.
    prev_end = lo
    for ti, s, e in acc:
        if prev_end < s:
            labels[prev_end:s] = "anticipatory"
            trial_id[prev_end:s] = ti
        labels[s:e] = "consummatory"
        trial_id[s:e] = ti
        prev_end = e
    # Licking after the final extension, out to the end of the session window.
    if last_e < hi:
        labels[last_e:hi] = "anticipatory"
        trial_id[last_e:hi] = acc[-1][0]
    labels[:lo] = "other"
    trial_id[:lo] = 0

    # Flag the licking that runs straight into an extension. Done before the post-reward carve
    # so that a lead-in beats it -- when trials come close together the tail of one reward and
    # the run into the next can overlap, and the run into the next is what matters.
    for ti, w in enumerate(windows, start=1):
        if w is None or not accepted[ti - 1]:
            continue
        a, b = w
        lead[a:b] = labels[a:b] == "anticipatory"

    # Reward clean-up licking is not anticipation, so carve it back out.
    if pr > 0:
        for ti, s, e in acc:
            stop = min(e + pr, n)
            seg = labels[e:stop]
            mask = (seg == "anticipatory") & ~lead[e:stop]
            seg[mask] = "post_reward"
            trial_id[e:stop][mask] = ti

    # A rejected extension still had the spout out, so its frames are not anticipatory.
    for i, (s, e) in enumerate(trials):
        if not accepted[i]:
            labels[s:e] = "other"
            trial_id[s:e] = 0
            lead[s:e] = False
    return labels, trial_id, lead


def extract_protrusions_and_bouts(df: pd.DataFrame, valid: np.ndarray, labels, trial_id,
                                  reg: SpoutRegions, cfg: dict, lead=None, trials=None,
                                  accepted=None):
    """Split clean tongue frames into individual protrusions, then merge protrusions into
    single-class bouts. A protrusion/bout never spans two labels or two trials, so licking
    that crosses an extension onset is split exactly at the onset.

    Every protrusion is reported, including one that sits entirely on its own. Bout membership
    is descriptive (`n_licks`, `isolated`), never a filter -- a lone lick is a bout of one.
    """
    frames = df["frame_idx"].to_numpy()
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return [], []
    if lead is None:
        lead = np.zeros(len(labels), dtype=bool)
    onset_of = {}
    if trials is not None:
        acc = accepted if accepted is not None else [True] * len(trials)
        onset_of = {i + 1: s for i, (s, _) in enumerate(trials) if acc[i]}

    tx = df["tongue.x"].to_numpy(float)
    ty = df["tongue.y"].to_numpy(float)
    ts = df["tongue.score"].to_numpy(float)
    sx = df["spout_tip.x"].to_numpy(float)
    sy = df["spout_tip.y"].to_numpy(float)

    lab_i, tid_i = labels[idx], trial_id[idx]

    # --- protrusions: consecutive frames, broken at any label/trial change ---
    brk = np.ones(len(idx), bool)
    brk[1:] = (np.diff(idx) > 1) | (lab_i[1:] != lab_i[:-1]) | (tid_i[1:] != tid_i[:-1])
    prot_groups = np.split(idx, np.where(brk)[0][1:])

    # --- bouts: merge protrusions within bout_gap_frames sharing label+trial ---
    bouts = []
    cur = []
    for g in prot_groups:
        if cur:
            prev = cur[-1]
            same = (labels[g[0]] == labels[prev[0]]) and (trial_id[g[0]] == trial_id[prev[0]])
            near = (frames[g[0]] - frames[prev[-1]]) <= cfg["bout_gap_frames"]
            if not (same and near):
                bouts.append(cur); cur = []
        cur.append(g)
    if cur:
        bouts.append(cur)

    pid = 0
    out_bouts, out_prots = [], []
    for bid, groups in enumerate(bouts, start=1):
        allf = np.concatenate(groups)
        i0, i1 = allf[0], allf[-1]
        win = np.arange(i0, i1 + 1)
        disp = _dist(sx[win], sy[win], reg.retracted[0], reg.retracted[1])
        out_bouts.append(Bout(
            bout_id=bid,
            trial_id=int(trial_id[i0]),
            classification=str(labels[i0]),
            start_frame=int(frames[i0]),
            end_frame=int(frames[i1]),
            n_licks=len(groups),
            n_tongue_frames=int(len(allf)),
            leads_into_reward=bool(lead[allf].any()),
            mean_spout_disp=(round(float(np.nanmean(disp)), 1)
                             if np.isfinite(disp).any() else float("nan")),
            mean_tongue_score=round(float(np.nanmean(ts[allf])), 3),
            tongue_x=round(float(np.nanmean(tx[allf])), 1),
            tongue_y=round(float(np.nanmean(ty[allf])), 1),
        ))
        for g in groups:
            pid += 1
            peak = g[int(np.nanargmax(ts[g]))]
            tid = int(trial_id[g[0]])
            onset = onset_of.get(tid)
            gap = ((onset - g[0]) / cfg["fps"]) if (onset is not None and onset > g[0]) \
                else float("nan")
            out_prots.append(Protrusion(
                protrusion_id=pid,
                trial_id=tid,
                bout_id=bid,
                label=str(labels[g[0]]),
                onset_frame=int(frames[g[0]]),
                peak_frame=int(frames[peak]),
                offset_frame=int(frames[g[-1]]),
                n_frames=int(len(g)),
                max_score=round(float(np.nanmax(ts[g])), 3),
                leads_into_reward=bool(lead[g].any()),
                isolated=bool(len(groups) == 1),
                s_to_extension=round(gap, 3) if gap == gap else float("nan"),
            ))
    return out_prots, out_bouts


# --------------------------------------------------------------------------------------
# Output assembly
# --------------------------------------------------------------------------------------
def bouts_to_frame(bouts, fps: float, offset_ms: float | None) -> pd.DataFrame:
    rows = []
    for b in bouts:
        start_s = b.start_frame / fps
        end_s = b.end_frame / fps
        row = asdict(b)
        row["start_s"] = round(start_s, 3)
        row["end_s"] = round(end_s, 3)
        row["duration_s"] = round((b.end_frame - b.start_frame + 1) / fps, 3)
        if offset_ms is not None:
            row["start_rig_ms"] = round(start_s * 1000.0 + offset_ms, 1)
            row["end_rig_ms"] = round(end_s * 1000.0 + offset_ms, 1)
        rows.append(row)
    cols = ["bout_id", "trial_id", "classification", "start_frame", "end_frame",
            "start_s", "end_s", "duration_s", "n_licks", "leads_into_reward",
            "n_tongue_frames", "mean_spout_disp", "mean_tongue_score", "tongue_x", "tongue_y"]
    if offset_ms is not None:
        cols += ["start_rig_ms", "end_rig_ms"]
    df = pd.DataFrame(rows)
    return df[[c for c in cols if c in df.columns]]


def protrusions_to_frame(prots, fps: float, offset_ms: float | None) -> pd.DataFrame:
    """One row per individual tongue protrusion -- the per-lick view that lines up with
    per-lick hand scoring."""
    rows = []
    for p in prots:
        row = asdict(p)
        row["onset_s"] = round(p.onset_frame / fps, 3)
        row["peak_s"] = round(p.peak_frame / fps, 3)
        row["duration_s"] = round((p.offset_frame - p.onset_frame + 1) / fps, 3)
        if offset_ms is not None:
            row["onset_rig_ms"] = round(p.onset_frame / fps * 1000.0 + offset_ms, 1)
        rows.append(row)
    cols = ["protrusion_id", "trial_id", "bout_id", "label", "onset_frame", "peak_frame",
            "offset_frame", "onset_s", "peak_s", "duration_s", "n_frames", "max_score",
            "leads_into_reward", "isolated", "s_to_extension"]
    if offset_ms is not None:
        cols += ["onset_rig_ms"]
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return pd.DataFrame(columns=cols)
    return df[[c for c in cols if c in df.columns]]


def trials_to_frame(trials, windows, frames, labels, trial_id, valid,
                    fps: float, offset_ms: float | None, assess=None) -> pd.DataFrame:
    """One row per spout-extension event, making the trial structure auditable."""
    rows = []
    lab_v = labels[valid]
    tid_v = trial_id[valid]
    if assess is None:
        assess = [dict(duration_s=0.0, duration_outlier=False, outside_session=False,
                       accepted=True) for _ in trials]
    for ti, ((s, e), w) in enumerate(zip(trials, windows), start=1):
        a = assess[ti - 1]
        n_anti = int(np.sum((tid_v == ti) & (lab_v == "anticipatory")))
        n_cons = int(np.sum((tid_v == ti) & (lab_v == "consummatory")))
        row = dict(
            trial_id=ti,
            accepted=a["accepted"],
            extension_onset_frame=int(frames[s]),
            extension_offset_frame=int(frames[min(e, len(frames) - 1)]),
            extension_onset_s=round(frames[s] / fps, 3),
            extension_offset_s=round(frames[min(e, len(frames) - 1)] / fps, 3),
            extension_duration_s=round((e - s) / fps, 3),
            duration_outlier=a["duration_outlier"],
            outside_session=a["outside_session"],
            has_anticipatory=w is not None,
            anticipatory_start_frame=int(frames[w[0]]) if w else None,
            anticipatory_start_s=round(frames[w[0]] / fps, 3) if w else None,
            anticipatory_duration_s=round((w[1] - w[0]) / fps, 3) if w else 0.0,
            n_anticipatory_lick_frames=n_anti,
            n_consummatory_lick_frames=n_cons,
        )
        if offset_ms is not None:
            row["extension_onset_rig_ms"] = round(frames[s] / fps * 1000.0 + offset_ms, 1)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(stem, reg, bouts_df, state, trials, windows):
    """Compact per-session rollup (one row) used only for the folder-mode combined CSV."""
    def n(c):
        return int((bouts_df["classification"] == c).sum()) if len(bouts_df) else 0
    return dict(
        video=stem,
        n_trials=len(trials),
        n_trials_with_anticipatory=int(sum(w is not None for w in windows)),
        n_bouts=int(len(bouts_df)),
        n_anticipatory=n("anticipatory"),
        n_consummatory=n("consummatory"),
        n_post_reward=n("post_reward"),
        n_other=n("other"),
        frac_frames_spout_extended=round(float(np.mean(state.to_numpy() == "extended")), 4),
        spout_region_separation_px=round(reg.separation, 1),
    )


# --------------------------------------------------------------------------------------
# Optional sensor verification
# --------------------------------------------------------------------------------------
def verify_with_sensor(bouts_df, sensor_path, cfg, fps, offset_ms,
                       code_col, time_col, lick_code):
    if offset_ms is None:
        print("  [sensor verification skipped: --video-rig-offset-ms not provided]")
        return None
    sensor = pd.read_csv(sensor_path)
    if code_col not in sensor.columns or time_col not in sensor.columns:
        print(f"  [sensor verification skipped: columns {code_col!r}/{time_col!r} "
              f"not in {list(sensor.columns)}]")
        return None
    contacts = sensor.loc[sensor[code_col] == lick_code, time_col].to_numpy(float)
    if "start_rig_ms" not in bouts_df.columns:
        print("  [sensor verification skipped: bout rig-ms not computed]")
        return None

    starts = bouts_df["start_rig_ms"].to_numpy(float)
    ends = bouts_df["end_rig_ms"].to_numpy(float)
    cls = bouts_df["classification"].to_numpy()
    tol = 1000.0 / fps * 1.5  # half-window slack of ~1.5 frames

    # which bout (if any) contains each contact?
    contact_hits = np.full(len(contacts), -1, dtype=int)
    for ci, t in enumerate(contacts):
        inside = np.where((t >= starts - tol) & (t <= ends + tol))[0]
        if len(inside):
            contact_hits[ci] = inside[0]

    bout_has_contact = np.zeros(len(bouts_df), dtype=bool)
    for bi in contact_hits[contact_hits >= 0]:
        bout_has_contact[bi] = True

    rep = dict(
        n_contacts=int(len(contacts)),
        contacts_in_consummatory=int(np.sum([cls[h] == "consummatory"
                                             for h in contact_hits if h >= 0])),
        contacts_in_anticipatory=int(np.sum([cls[h] == "anticipatory"
                                             for h in contact_hits if h >= 0])),
        contacts_in_ambiguous=int(np.sum([cls[h] == "ambiguous"
                                          for h in contact_hits if h >= 0])),
        contacts_in_no_bout=int(np.sum(contact_hits < 0)),
        consummatory_bouts_with_contact=int(np.sum(bout_has_contact & (cls == "consummatory"))),
        consummatory_bouts_without_contact=int(np.sum(~bout_has_contact & (cls == "consummatory"))),
        anticipatory_bouts_with_contact=int(np.sum(bout_has_contact & (cls == "anticipatory"))),
        anticipatory_bouts_without_contact=int(np.sum(~bout_has_contact & (cls == "anticipatory"))),
    )
    print("  sensor verification:")
    for k, v in rep.items():
        print(f"    {k}: {v}")
    if rep["anticipatory_bouts_with_contact"] > 0:
        print("    NOTE: anticipatory bouts containing a contact indicate possible "
              "misclassification or alignment offset error.")
    return rep


# --------------------------------------------------------------------------------------
# Per-file driver
# --------------------------------------------------------------------------------------
def process_one(csv_path: str, args, cfg: dict, out_dir: str) -> dict:
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    print(f"\n=== {stem} ===")
    df = load_sleap_csv(csv_path)

    reg = learn_spout_regions(df, cfg,
                              ret_override=args.retracted_region,
                              ext_override=args.extended_region)
    print(f"  retracted region: ({reg.retracted[0]:.1f},{reg.retracted[1]:.1f})  "
          f"extended region: "
          f"{'none' if reg.extended is None else f'({reg.extended[0]:.1f},{reg.extended[1]:.1f})'}  "
          f"sep={reg.separation:.1f}px  [{reg.note}]")
    if reg.extended is None:
        print("  WARNING: no extended spout region found -> no reward trials, so there are no "
              "anticipatory windows; all licks are labelled 'other'.")

    trials = find_extension_events(df, reg, cfg)
    state = state_from_trials(len(df), trials, index=df.index)
    valid, tongue_info = clean_tongue(df, cfg)
    roi_cut = tongue_info["roi_cut_px"]
    print(f"  tongue: {tongue_info['n_score_pass']} score-pass, "
          f"{tongue_info['n_roi_rejected']} ROI-rejected, "
          f"{tongue_info['n_valid']} valid frames "
          f"(ROI cut {roi_cut:.1f}px)" if roi_cut else
          f"  tongue: {tongue_info['n_valid']} valid frames (no ROI learned)")
    if tongue_info["n_nose_missing"]:
        print(f"          ({tongue_info['n_nose_missing']} frames lack a nose; "
              f"session-median nose used as the ROI anchor there)")

    assess = assess_trials(trials, cfg, len(df))
    accepted = [a["accepted"] for a in assess]
    if args.keep_all_trials:
        accepted = [True] * len(trials)
    windows = derive_anticipatory_windows(valid, trials, cfg, accepted)
    bounds = session_bounds(df, reg, trials, accepted, cfg, len(df))
    labels, trial_id, lead = label_licks(len(df), trials, windows, cfg, accepted, bounds)
    prots, bouts = extract_protrusions_and_bouts(df, valid, labels, trial_id, reg, cfg,
                                                 lead, trials, accepted)

    bouts_df = bouts_to_frame(bouts, cfg["fps"], args.video_rig_offset_ms)
    prots_df = protrusions_to_frame(prots, cfg["fps"], args.video_rig_offset_ms)
    frames_arr = df["frame_idx"].to_numpy()
    trials_df = trials_to_frame(trials, windows, frames_arr, labels, trial_id, valid,
                                cfg["fps"], args.video_rig_offset_ms, assess)

    n_ok = int(sum(accepted))
    n_with = int(sum(w is not None for w in windows))
    print(f"  trials (spout extensions): {len(trials)} detected, {n_ok} accepted as reward "
          f"trials ({n_with} of those with a licking lead-in)")
    rejected = [(i + 1, a) for i, a in enumerate(assess) if not accepted[i]]
    if rejected:
        med = float(np.median([a["duration_s"] for a in assess]))
        print(f"  excluded {len(rejected)} extension(s) as edge-of-recording / off-protocol "
              f"artifacts (session median {med:.2f}s):")
        for ti, a in rejected:
            why = ", ".join(([f"duration {a['duration_s']:.2f}s"] if a["duration_outlier"] else [])
                            + (["outside active session"] if a["outside_session"] else []))
            t0 = trials_df.loc[trials_df.trial_id == ti, "extension_onset_s"].iloc[0]
            print(f"        trial {ti} at t={t0:.1f}s -- {why}")
        print("        (their licks are labelled 'other'; use --keep-all-trials to include them)")
    counts = {c: int((bouts_df["classification"] == c).sum()) if len(bouts_df) else 0
              for c in ("anticipatory", "consummatory", "post_reward", "other")}
    print(f"  bouts: {len(bouts_df)} total -> {counts['anticipatory']} anticipatory, "
          f"{counts['consummatory']} consummatory, {counts['post_reward']} post-reward, "
          f"{counts['other']} other")
    if len(prots_df):
        ap = prots_df[prots_df["label"] == "anticipatory"]
        if len(ap):
            lead_n = int(ap["leads_into_reward"].sum())
            spont = ap[~ap["leads_into_reward"].astype(bool)]
            solo = int(spont["isolated"].sum())
            print(f"  anticipatory licks: {len(ap)} total -> {lead_n} in a run leading into a "
                  f"reward, {len(spont)} spontaneous between trials "
                  f"({solo} of those a lone single lick)")

    # --- write outputs ---
    base = os.path.join(out_dir, stem)
    anti_p = prots_df[prots_df["label"] == "anticipatory"] if len(prots_df) else prots_df
    anti_b = bouts_df[bouts_df["classification"] == "anticipatory"] if len(bouts_df) else bouts_df
    anti_p.drop(columns=["label"], errors="ignore").to_csv(
        base + "_anticipatory_licks.csv", index=False)
    anti_b.drop(columns=["classification"], errors="ignore").to_csv(
        base + "_anticipatory_bouts.csv", index=False)
    bouts_df.to_csv(base + "_lick_bouts.csv", index=False)
    trials_df.to_csv(base + "_trials.csv", index=False)

    if args.emit_frames:
        ann = df.copy()
        ann["tongue_valid"] = valid
        ann["spout_state"] = state.values
        ann["lick_label"] = labels
        ann["trial_id"] = trial_id
        ann["spout_disp"] = _dist(df["spout_tip.x"], df["spout_tip.y"],
                                  reg.retracted[0], reg.retracted[1])
        ann.to_csv(base + "_frames_annotated.csv", index=False)

    if args.sensor_log:
        verify_with_sensor(bouts_df, args.sensor_log, cfg, cfg["fps"],
                           args.video_rig_offset_ms, args.sensor_code_col,
                           args.sensor_time_col, args.lick_code)

    print(f"  wrote _anticipatory_licks.csv ({len(anti_p)} individual licks), "
          f"_anticipatory_bouts.csv ({len(anti_b)}), _lick_bouts.csv, _trials.csv")
    return summarize(stem, reg, bouts_df, state, trials, windows)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def _pair(s):
    x, y = s.split(",")
    return (float(x), float(y))


def build_argparser():
    p = argparse.ArgumentParser(
        description="Detect anticipatory vs consummatory licks from a SLEAP analysis CSV. "
                    "The CSV is the ONLY input required -- no video, no .slp, no rig log. "
                    "Everything else is an optional override.",
        epilog="Needs only pandas and numpy.")
    p.add_argument("input", help="a SLEAP analysis CSV, or a folder of them. This is the only "
                                 "input the script needs")
    p.add_argument("--out-dir", default=None,
                   help="output directory (default: alongside each input CSV)")
    p.add_argument("--fps", type=float, default=DEFAULTS["fps"])
    p.add_argument("--spout-score-min", type=float, default=DEFAULTS["spout_score_min"])
    p.add_argument("--tongue-score-min", type=float, default=DEFAULTS["tongue_score_min"])
    p.add_argument("--tongue-roi-k", type=float, default=DEFAULTS["tongue_roi_k"])
    p.add_argument("--min-protrusion-frames", type=int, default=DEFAULTS["min_protrusion_frames"])
    p.add_argument("--bout-gap-frames", type=int, default=DEFAULTS["bout_gap_frames"])
    p.add_argument("--min-extended-run", type=int, default=DEFAULTS["min_extended_run"],
                   help="min frames for a real spout extension (OPEN)")
    p.add_argument("--extension-merge-gap", type=int, default=DEFAULTS["extension_merge_gap"],
                   help="bridge extended runs this many frames apart, fixing detector flicker (CLOSE)")
    p.add_argument("--lick-episode-gap", type=int, default=DEFAULTS["lick_episode_gap"],
                   help="licks within this many frames are one episode; defines 'when the "
                        "mouse starts licking' (default 120 = 2s @60fps)")
    p.add_argument("--anticipatory-max-gap", type=int, default=DEFAULTS["anticipatory_max_gap"],
                   help="max frames from the last lick to the extension onset for the trial "
                        "to count as having an anticipatory lead-in")
    p.add_argument("--post-reward-s", type=float, default=DEFAULTS["post_reward_s"],
                   help="licking within this many seconds after the spout retracts is reward "
                        "clean-up rather than anticipation. Use 0 to count it as anticipatory")
    p.add_argument("--min-extension-s", type=float, default=None,
                   help="drop extensions shorter than this (s) -- removes blips")
    p.add_argument("--max-extension-s", type=float, default=None,
                   help="drop extensions longer than this (s) -- e.g. the spout parked out "
                        "before the session starts")
    p.add_argument("--trim-start-s", type=float, default=None,
                   help="ignore extensions in the first N seconds (mouse being put on the rig). "
                        "Default: auto -- the active session is inferred from well-formed trials")
    p.add_argument("--trim-end-s", type=float, default=None,
                   help="ignore extensions in the last N seconds (mouse being taken off)")
    p.add_argument("--keep-all-trials", action="store_true",
                   help="do not exclude edge-of-recording / off-protocol extensions")
    # region overrides (for cameras where auto-calibration struggles)
    p.add_argument("--retracted-region", type=_pair, default=None,
                   help="override retracted centroid as 'x,y'")
    p.add_argument("--extended-region", type=_pair, default=None,
                   help="override extended centroid as 'x,y'")
    # sensor verification
    p.add_argument("--sensor-log", default=None, help="rig touch-sensor CSV for verification")
    p.add_argument("--video-rig-offset-ms", type=float, default=None,
                   help="rig_ms = frame/fps*1000 + offset (from your alignment protocol)")
    p.add_argument("--sensor-code-col", default="code")
    p.add_argument("--sensor-time-col", default="timestamp_ms")
    p.add_argument("--lick-code", default=None, help="event code value that denotes a contact")
    # output toggles
    p.add_argument("--emit-frames", action="store_true",
                   help="also write the per-frame annotated CSV")
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)

    cfg = dict(DEFAULTS)
    cfg.update(fps=args.fps, spout_score_min=args.spout_score_min,
               tongue_score_min=args.tongue_score_min, tongue_roi_k=args.tongue_roi_k,
               min_protrusion_frames=args.min_protrusion_frames,
               bout_gap_frames=args.bout_gap_frames, min_extended_run=args.min_extended_run,
               extension_merge_gap=args.extension_merge_gap,
               lick_episode_gap=args.lick_episode_gap,
               anticipatory_max_gap=args.anticipatory_max_gap,
               post_reward_s=args.post_reward_s,
               min_extension_s=args.min_extension_s,
               max_extension_s=args.max_extension_s,
               trim_start_s=args.trim_start_s, trim_end_s=args.trim_end_s)
    # try to coerce lick_code to int if it looks numeric
    if args.lick_code is not None:
        try:
            args.lick_code = int(args.lick_code)
        except ValueError:
            pass

    if os.path.isdir(args.input):
        files = find_input_csvs(args.input)
        if not files:
            print(f"No SLEAP analysis CSVs found in {args.input}")
            return 1
    else:
        files = [args.input]

    summaries, failed = [], []
    for f in files:
        out_dir = args.out_dir or os.path.dirname(os.path.abspath(f))
        os.makedirs(out_dir, exist_ok=True)
        try:
            summaries.append(process_one(f, args, cfg, out_dir))
        except Exception as e:
            failed.append((f, e))
            print(f"  ERROR processing {os.path.basename(f)}: {e}")

    if len(files) > 1 and summaries:
        out_dir = args.out_dir or os.path.dirname(os.path.abspath(files[0]))
        combined = pd.DataFrame(summaries)
        combined_path = os.path.join(out_dir, "combined_session_summary.csv")
        combined.to_csv(combined_path, index=False)
        print(f"\nWrote combined summary across {len(summaries)} sessions: {combined_path}")
        print(combined.to_string(index=False))

    # Report failure through the exit code as well as the log, so a broken CSV cannot slip
    # through a batch run unnoticed.
    if failed:
        print(f"\n{len(failed)} of {len(files)} file(s) could not be processed.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
