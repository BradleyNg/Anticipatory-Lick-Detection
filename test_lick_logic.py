#!/usr/bin/env python3
"""
test_lick_logic.py
==================

Tests for the trial-based anticipatory-lick logic in detect_anticipatory_licks.py.

Each synthetic case builds a fake SLEAP frame table with a known ground truth and asserts the
exact labelling. A final regression test runs the real sample CSV (if present) and pins the
measured numbers.

Run:  python test_lick_logic.py
"""

from __future__ import annotations

import glob
import sys
import traceback

import numpy as np
import pandas as pd

from detect_anticipatory_licks import (
    DEFAULTS, load_sleap_csv, learn_spout_regions, find_extension_events,
    derive_anticipatory_windows, label_licks, clean_tongue,
    extract_protrusions_and_bouts, assess_trials,
)

RETRACTED = (100.0, 100.0)
EXTENDED = (200.0, 100.0)


def make_df(n, extensions=(), tongue_runs=(), tongue_xy=(150.0, 90.0)):
    """Synthetic SLEAP table: spout parked at RETRACTED except during `extensions`;
    tongue present (score 0.9) during `tongue_runs`."""
    rng = np.random.default_rng(0)
    sx = np.full(n, RETRACTED[0]) + rng.normal(0, 0.05, n)
    sy = np.full(n, RETRACTED[1]) + rng.normal(0, 0.05, n)
    for s, e in extensions:
        sx[s:e] = EXTENDED[0] + rng.normal(0, 0.05, e - s)
        sy[s:e] = EXTENDED[1] + rng.normal(0, 0.05, e - s)
    tx = np.full(n, np.nan)
    ty = np.full(n, np.nan)
    tsc = np.zeros(n)
    for s, e in tongue_runs:
        tx[s:e] = tongue_xy[0]
        ty[s:e] = tongue_xy[1]
        tsc[s:e] = 0.9
    return pd.DataFrame({
        "track": [np.nan] * n, "frame_idx": np.arange(n), "instance.score": 1.0,
        "left_paw.score": 1.0, "left_paw.x": 50.0, "left_paw.y": 200.0,
        "nose.score": 1.0, "nose.x": 150.0, "nose.y": 60.0,
        "right_paw.score": 1.0, "right_paw.x": 250.0, "right_paw.y": 200.0,
        "spout_tip.score": 1.0, "spout_tip.x": sx, "spout_tip.y": sy,
        "tongue.score": tsc, "tongue.x": tx, "tongue.y": ty,
    })


def run_pipeline(df, keep_all=False, **over):
    cfg = dict(DEFAULTS)
    cfg.update(over)
    reg = learn_spout_regions(df, cfg)
    trials = find_extension_events(df, reg, cfg)
    valid, _ = clean_tongue(df, cfg)
    assess = assess_trials(trials, cfg, len(df))
    accepted = [True] * len(trials) if keep_all else [a["accepted"] for a in assess]
    windows = derive_anticipatory_windows(valid, trials, cfg, accepted)
    labels, trial_id, lead = label_licks(len(df), trials, windows, cfg, accepted)
    prots, bouts = extract_protrusions_and_bouts(df, valid, labels, trial_id, reg, cfg,
                                                 lead, trials, accepted)
    return dict(reg=reg, trials=trials, valid=valid, windows=windows, assess=assess,
                accepted=accepted, labels=labels, trial_id=trial_id, lead=lead,
                prots=prots, bouts=bouts)


def labels_of(r, s, e):
    """Set of labels assigned to the clean lick frames in [s, e)."""
    sel = r["valid"][s:e]
    return set(np.asarray(r["labels"][s:e])[sel].tolist())


# ---------------------------------------------------------------------------- tests
def test_leadin():
    """Licking that runs into an extension is anticipatory; licking during it is consummatory."""
    df = make_df(3000, extensions=[(1000, 1180)],
                 tongue_runs=[(900, 910), (930, 940), (960, 970), (1020, 1040)])
    r = run_pipeline(df)
    assert len(r["trials"]) == 1, r["trials"]
    assert r["windows"][0] is not None
    assert r["windows"][0] == (900, 1000), r["windows"][0]
    assert labels_of(r, 900, 1000) == {"anticipatory"}
    assert labels_of(r, 1020, 1040) == {"consummatory"}
    assert all(p.trial_id == 1 for p in r["prots"])


def test_no_leadin():
    """A trial the mouse did not lick into has no lead-in run, but an earlier lone lick is
    still an anticipatory lick. It just is not flagged as leading into the reward."""
    df = make_df(3000, extensions=[(1000, 1180)],
                 tongue_runs=[(100, 110), (1020, 1040)])   # 100-110 is far before onset
    r = run_pipeline(df)
    assert r["windows"] == [None], r["windows"]
    assert labels_of(r, 100, 110) == {"anticipatory"}
    assert labels_of(r, 1020, 1040) == {"consummatory"}
    early = [p for p in r["prots"] if p.onset_frame == 100]
    assert len(early) == 1 and early[0].leads_into_reward is False


def test_isolated_single_lick_is_kept():
    """A lone lick made between trials with the spout retracted is still anticipatory.

    The mouse often licks once without working up to a proper run of licking. Requiring an
    'established' multi-lick bout would throw those away, so bout size must never be a filter.
    """
    df = make_df(6000, extensions=[(1000, 1180), (4000, 4180)],
                 tongue_runs=[(2500, 2504)])            # one lick, far from either extension
    r = run_pipeline(df)
    assert labels_of(r, 2500, 2504) == {"anticipatory"}
    anti = [p for p in r["prots"] if p.label == "anticipatory"]
    assert len(anti) == 1, anti
    p = anti[0]
    assert p.onset_frame == 2500
    assert p.isolated is True                            # a bout of one
    assert p.leads_into_reward is False                  # spontaneous, not a lead-in
    assert p.trial_id == 2                               # attributed to the trial it precedes
    assert abs(p.s_to_extension - 25.0) < 0.05           # 1500 frames at 60fps


def test_many_isolated_licks_all_survive():
    """Several scattered single licks are each reported, not collapsed or dropped."""
    singles = [(300, 304), (900, 904), (2000, 2004), (2600, 2604), (3300, 3304)]
    df = make_df(6000, extensions=[(4000, 4180)], tongue_runs=singles)
    r = run_pipeline(df)
    anti = sorted(p.onset_frame for p in r["prots"] if p.label == "anticipatory")
    assert anti == [s for s, _ in singles], anti
    assert all(p.isolated for p in r["prots"] if p.label == "anticipatory")


def test_lead_in_flag_separates_runs_from_singles():
    """leads_into_reward marks the run that builds into the extension, without excluding
    the spontaneous licks from the anticipatory output."""
    df = make_df(6000, extensions=[(4000, 4180)],
                 tongue_runs=[(2000, 2004),                         # spontaneous single
                              (3900, 3910), (3930, 3940)])          # run into the extension
    r = run_pipeline(df)
    anti = {p.onset_frame: p for p in r["prots"] if p.label == "anticipatory"}
    assert set(anti) == {2000, 3900, 3930}, sorted(anti)
    assert anti[2000].leads_into_reward is False
    assert anti[3900].leads_into_reward is True
    assert anti[3930].leads_into_reward is True


def test_span_extension_onset():
    """A single lick train crossing the onset splits exactly at the onset."""
    df = make_df(3000, extensions=[(1000, 1180)], tongue_runs=[(990, 1010)])
    r = run_pipeline(df)
    assert labels_of(r, 990, 1000) == {"anticipatory"}
    assert labels_of(r, 1000, 1010) == {"consummatory"}
    onsets = sorted(p.onset_frame for p in r["prots"])
    assert onsets == [990, 1000], onsets          # split into two protrusions
    assert len(r["bouts"]) == 2


def test_post_retraction_not_absorbed():
    """Licking belonging to the previous trial must not be pulled into the next trial's
    anticipatory window (clip-to-previous-offset)."""
    df = make_df(3000, extensions=[(500, 680), (800, 980)],
                 tongue_runs=[(600, 620),      # during trial 1 -> consummatory
                              (790, 795)])     # genuine lead-in for trial 2
    r = run_pipeline(df)
    assert len(r["trials"]) == 2
    assert labels_of(r, 600, 620) == {"consummatory"}
    # window for trial 2 must start at 790, NOT reach back into trial 1
    assert r["windows"][1] == (790, 800), r["windows"][1]
    assert labels_of(r, 790, 795) == {"anticipatory"}


def test_post_reward_licking_separated():
    """Licking immediately after the spout retracts is reward clean-up, not anticipation,
    but licking later in the same gap is anticipatory again."""
    df = make_df(4000, extensions=[(500, 680), (2000, 2180)],
                 tongue_runs=[(685, 700),      # right after retraction -> post_reward
                              (1200, 1204)])   # well into the gap -> anticipatory
    r = run_pipeline(df)
    assert labels_of(r, 685, 700) == {"post_reward"}
    assert labels_of(r, 1200, 1204) == {"anticipatory"}
    # with the window switched off it counts as anticipatory instead
    r0 = run_pipeline(df, post_reward_s=0.0)
    assert labels_of(r0, 685, 700) == {"anticipatory"}


def test_flicker_fragments_merge_to_one_trial():
    """Detector flicker splitting one extension into fragments must yield ONE trial."""
    df = make_df(3000, extensions=[(1000, 1010), (1012, 1020), (1025, 1180)])
    r = run_pipeline(df)
    assert len(r["trials"]) == 1, r["trials"]
    assert r["trials"][0] == (1000, 1180), r["trials"][0]


def test_short_blip_dropped():
    """A sub-threshold extended blip is not a trial."""
    df = make_df(3000, extensions=[(1000, 1003), (2000, 2180)])
    r = run_pipeline(df)
    assert len(r["trials"]) == 1, r["trials"]
    assert r["trials"][0] == (2000, 2180)


def test_no_extensions_all_other():
    """No spout extension anywhere -> no trials, no anticipatory licks."""
    df = make_df(3000, extensions=[], tongue_runs=[(500, 520), (900, 920)])
    r = run_pipeline(df)
    assert r["trials"] == []
    assert labels_of(r, 500, 520) == {"other"}
    assert labels_of(r, 900, 920) == {"other"}
    assert not any(p.label == "anticipatory" for p in r["prots"])


def test_protrusion_counting():
    """N separate tongue runs close together = 1 bout with N individual licks."""
    df = make_df(3000, extensions=[(1000, 1180)],
                 tongue_runs=[(900, 905), (912, 917), (924, 929)])
    r = run_pipeline(df)
    anti_bouts = [b for b in r["bouts"] if b.classification == "anticipatory"]
    assert len(anti_bouts) == 1, anti_bouts
    assert anti_bouts[0].n_licks == 3, anti_bouts[0]
    anti_prots = [p for p in r["prots"] if p.label == "anticipatory"]
    assert len(anti_prots) == 3
    assert [p.onset_frame for p in anti_prots] == [900, 912, 924]


def test_edge_of_recording_artifact_excluded():
    """Raw recordings start/end with the mouse being put on / taken off the rig. A long
    spout-parked-out period at the start is off-protocol AND outside the active session,
    so it must not be treated as a reward trial -- its licks are 'other', not consummatory."""
    df = make_df(6000,
                 extensions=[(100, 1900),                       # 30s parked-out artifact
                             (2500, 2680), (3500, 3680), (4500, 4680)],
                 tongue_runs=[(600, 620),                       # licking during the artifact
                              (2480, 2495)])                    # real lead-in for trial 2
    r = run_pipeline(df)
    a = r["assess"]
    assert a[0]["duration_outlier"] and a[0]["outside_session"] and not a[0]["accepted"]
    assert all(x["accepted"] for x in a[1:]), a
    assert labels_of(r, 600, 620) == {"other"}          # NOT consummatory
    assert labels_of(r, 2480, 2495) == {"anticipatory"}
    assert r["windows"][0] is None                       # rejected trial earns no window
    # --keep-all-trials restores the old behaviour
    r2 = run_pipeline(df, keep_all=True)
    assert labels_of(r2, 600, 620) == {"consummatory"}


def test_trailing_blips_outside_session_excluded():
    """Sub-second 'extensions' after the last real trial (mouse coming off the rig) are
    excluded rather than generating spurious anticipatory licks."""
    df = make_df(9000,
                 extensions=[(1000, 1180), (2000, 2180), (3000, 3180),
                             (8000, 8008), (8500, 8506)],       # trailing blips (survive OPEN)
                 tongue_runs=[(7950, 7995)])                    # licking before a blip
    r = run_pipeline(df)
    assert [x["accepted"] for x in r["assess"]] == [True, True, True, False, False]
    assert labels_of(r, 7950, 7995) == {"other"}
    assert not any(p.label == "anticipatory" for p in r["prots"])


def test_explicit_trim_overrides():
    """--trim-start-s / --trim-end-s explicitly bound the usable session."""
    df = make_df(9000, extensions=[(1000, 1180), (5000, 5180), (8000, 8180)])
    r = run_pipeline(df, trim_start_s=30.0)      # 30s = frame 1800 -> drops trial 1
    assert [x["accepted"] for x in r["assess"]] == [False, True, True]
    r2 = run_pipeline(df, trim_end_s=30.0)       # last 30s -> frame 7200 -> drops trial 3
    assert [x["accepted"] for x in r2["assess"]] == [True, True, False]


NODES = ["nose", "tongue", "spout_tip", "left_paw", "right_paw"]


def warp(df, ang=0.0, sx=1.0, sy=1.0, tx=0.0, ty=0.0, mirror=False, M=None, persp=None):
    """Re-project every tracked point, simulating a different camera pose."""
    d = df.copy()
    th = np.radians(ang)
    c, s = np.cos(th), np.sin(th)
    for n in NODES:
        x = d[f"{n}.x"].to_numpy(float).copy()
        y = d[f"{n}.y"].to_numpy(float).copy()
        if mirror:
            x = -x
        if M is not None:
            X = M[0][0] * x + M[0][1] * y + M[0][2]
            Y = M[1][0] * x + M[1][1] * y + M[1][2]
        else:
            X = c * (x * sx) - s * (y * sy) + tx
            Y = s * (x * sx) + c * (y * sy) + ty
        if persp is not None:
            p, q = persp
            w = 1.0 + p * X + q * Y
            X, Y = X / w, Y / w
        d[f"{n}.x"], d[f"{n}.y"] = X, Y
    return d


def _fingerprint(r):
    """What must be identical regardless of camera pose."""
    lab = np.asarray(r["labels"])[r["valid"]]
    return dict(
        accepted_trials=[tuple(t) for t, ok in zip(r["trials"], r["accepted"]) if ok],
        counts={k: int((lab == k).sum()) for k in ("anticipatory", "consummatory", "other")},
        anti_onsets=[p.onset_frame for p in r["prots"] if p.label == "anticipatory"],
    )


def test_camera_pose_invariance():
    """The spout extended/retracted decision must be RELATIVE, not tied to absolute x,y.

    Every video is a different camera angle/zoom, so re-projecting all tracked points must
    not change which licks are anticipatory. Note we compare the *accepted* trials and the
    labelling -- a rejected sub-second artifact can legitimately fall below the detection
    gate under a strong distortion, which does not affect any real result.
    """
    files = glob.glob("*analysis*.csv")
    if not files:
        print("    (skipped: no sample CSV present)")
        return
    df = load_sleap_csv(files[0])
    base = _fingerprint(run_pipeline(df))
    shear_a = [[1, 0.45, 0], [0.2, 1, 0]]
    shear_b = [[1, -0.6, 0], [0.35, 1, 0]]
    cases = {
        "translate": dict(tx=500, ty=-300),
        "rotate 37deg": dict(ang=37),
        "rotate 113deg": dict(ang=113),
        "scale 0.6": dict(sx=0.6, sy=0.6),
        "scale 2.5": dict(sx=2.5, sy=2.5),
        "mirror": dict(mirror=True),
        "rot+scale+translate": dict(ang=25, sx=1.8, sy=1.8, tx=-120, ty=640),
        "anisotropic": dict(sx=1.5, sy=0.8),
        "shear A": dict(M=shear_a),
        "shear B": dict(M=shear_b),
        "perspective": dict(persp=(3e-4, -2e-4)),
        "perspective strong": dict(persp=(8e-4, 5e-4)),
        "shear+perspective": dict(M=shear_a, persp=(4e-4, -3e-4)),
    }
    seps = []
    total = sum(base["counts"].values())
    for name, kw in cases.items():
        r = run_pipeline(warp(df, **kw))
        got = _fingerprint(r)
        # The deliverable must be EXACTLY reproduced.
        assert got["anti_onsets"] == base["anti_onsets"], f"{name}: anticipatory licks changed"
        assert len(got["accepted_trials"]) == len(base["accepted_trials"]), \
            f"{name}: trial count changed"
        # Trial edges may move by a frame: the spout takes ~4 frames to travel, so the exact
        # transition frame is genuinely ambiguous and a re-projection can tip it either way.
        shift = max(abs(x[0] - y[0]) for x, y in
                    zip(base["accepted_trials"], got["accepted_trials"]))
        assert shift <= 2, f"{name}: trial onset moved {shift} frames"
        delta = max(abs(got["counts"][k] - base["counts"][k]) for k in base["counts"])
        assert delta <= max(5, 0.01 * total), f"{name}: labels shifted by {delta} frames"
        seps.append(r["reg"].separation)
    # the learned separation must actually track the zoom -- proving it is learned, not fixed
    assert max(seps) / min(seps) > 3, seps
    print(f"    {len(cases)} camera poses -> identical anticipatory licks; learned separation "
          f"ranged {min(seps):.1f}-{max(seps):.1f}px")


# Pinned results per known SLEAP export of video 1. Keyed by the model-run stamp in the
# filename, so whichever CSV(s) are present get checked against the right expectation.
EXPECTED = {
    "260618_082055": dict(  # first tongue-tracking model run
        ret=(296.1, 336.0), ext=(356.0, 304.0), detected=17, accepted=14, rejects=[1, 16, 17],
        leadin=13, window_median=193,
        counts={"anticipatory": 875, "consummatory": 1019, "post_reward": 65, "other": 533},
        anti_prot=184, anti_lead=129, anti_solo=46),
    "260723_143602": dict(  # retrained model -- fewer, cleaner tongue detections
        ret=(296.0, 336.0), ext=(352.0, 304.0), detected=16, accepted=14, rejects=[1, 16],
        leadin=12, window_median=192,
        counts={"anticipatory": 488, "consummatory": 766, "post_reward": 59, "other": 196},
        anti_prot=132, anti_lead=112, anti_solo=18),
}


def test_regression_real_csv():
    """Pin the measured numbers on every known real CSV present in the working directory."""
    files = [f for f in glob.glob("*analysis*.csv")
             if any(k in f for k in EXPECTED)]
    if not files:
        print("    (skipped: no known sample CSV present)")
        return
    for f in files:
        key = next(k for k in EXPECTED if k in f)
        exp = EXPECTED[key]
        r = run_pipeline(load_sleap_csv(f))
        reg = r["reg"]
        assert abs(reg.retracted[0] - exp["ret"][0]) < 1 and abs(reg.retracted[1] - exp["ret"][1]) < 1
        assert abs(reg.extended[0] - exp["ext"][0]) < 1 and abs(reg.extended[1] - exp["ext"][1]) < 1
        assert len(r["trials"]) == exp["detected"], (key, len(r["trials"]))
        assert sum(r["accepted"]) == exp["accepted"], (key, sum(r["accepted"]))
        rej = [i + 1 for i, ok in enumerate(r["accepted"]) if not ok]
        assert rej == exp["rejects"], (key, rej)
        # every reject is both off-protocol in duration AND outside the active session
        assert all(r["assess"][i - 1]["duration_outlier"] and
                   r["assess"][i - 1]["outside_session"] for i in rej), key
        assert sum(w is not None for w in r["windows"]) == exp["leadin"], key
        lab = np.asarray(r["labels"])[r["valid"]]
        counts = {k: int((lab == k).sum())
                  for k in ("anticipatory", "consummatory", "post_reward", "other")}
        assert counts == exp["counts"], (key, counts)
        wl = [b - a for w in r["windows"] if w for a, b in [w]]
        assert int(np.median(wl)) == exp["window_median"], (key, np.median(wl))
        ap = [p for p in r["prots"] if p.label == "anticipatory"]
        assert len(ap) == exp["anti_prot"], (key, len(ap))
        lead = sum(p.leads_into_reward for p in ap)
        solo = sum(p.isolated and not p.leads_into_reward for p in ap)
        assert lead == exp["anti_lead"], (key, lead)
        assert solo == exp["anti_solo"], (key, solo)
        print(f"    [{key}] {exp['detected']} detected / {exp['accepted']} accepted "
              f"({exp['leadin']} lead-in), {counts}, {len(ap)} anticipatory licks "
              f"({lead} lead-in, {solo} lone singles)")


# ---------------------------------------------------------------------------- runner
def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
