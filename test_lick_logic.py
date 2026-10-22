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
import os
import sys
import traceback

import numpy as np
import pandas as pd

from detect_anticipatory_licks import (
    DEFAULTS, load_sleap_csv, learn_spout_regions, find_extension_events,
    derive_anticipatory_windows, label_licks, clean_tongue,
    extract_protrusions_and_bouts, assess_trials, session_bounds, is_output_file,
    FRAME_PARAMS, resolve_frame_params,
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
    # The synthetic sessions and the three real exports are 60 fps; the script defaults to 30.
    cfg = resolve_frame_params(dict(DEFAULTS, fps=60.0))
    cfg.update(over)
    reg = learn_spout_regions(df, cfg)
    trials = find_extension_events(df, reg, cfg)
    valid, _ = clean_tongue(df, cfg)
    assess = assess_trials(trials, cfg, len(df))
    accepted = [True] * len(trials) if keep_all else [a["accepted"] for a in assess]
    windows = derive_anticipatory_windows(valid, trials, cfg, accepted)
    bounds = session_bounds(df, reg, trials, accepted, cfg, len(df))
    labels, trial_id, lead = label_licks(len(df), trials, windows, cfg, accepted, bounds)
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


def test_short_gap_between_rewards_keeps_anticipatory_licks():
    """The spout can retract and come straight back out, and the licks in that short gap are
    anticipating the next reward rather than finishing the last one.

    Two separate things protect them. Licking that runs into the next extension is flagged as a
    lead-in and beats the clean-up carve, which covers the gap whenever the lead-in reach is as
    long as the clean-up window. The carve is additionally capped at a share of the distance to
    the next extension, which is what protects a lick when the lead-in reach is configured
    shorter than the clean-up window.
    """
    # the clean-up window is off by default; these checks are about what it may claim when on
    on = dict(post_reward_s=2.0)
    # two real rewards 2 s apart, shorter than post_reward_s, with licking through the gap
    df = make_df(20000, extensions=[(2000, 2180), (2240, 2420), (10000, 10180)],
                 tongue_runs=[(2200, 2206), (2225, 2231)])
    r = run_pipeline(df, **on)
    assert len(r["trials"]) == 3, r["trials"]
    assert labels_of(r, 2225, 2231) == {"anticipatory"}, labels_of(r, 2225, 2231)

    # with a short lead-in reach the cap is the only thing standing between a mid-gap lick and
    # being written off as clean-up
    df2 = make_df(20000, extensions=[(2000, 2180), (2360, 2540), (10000, 10180)],
                  tongue_runs=[(2280, 2286)])
    kw = dict(anticipatory_max_gap=30, **on)
    assert labels_of(run_pipeline(df2, **kw), 2280, 2286) == {"anticipatory"}
    assert labels_of(run_pipeline(df2, post_reward_max_frac=1.0, **kw),
                     2280, 2286) == {"post_reward"}

    # a long gap is unaffected, clean-up still gets its full window
    df3 = make_df(20000, extensions=[(2000, 2180), (12000, 12180)],
                  tongue_runs=[(2200, 2206), (2400, 2406)])
    r3 = run_pipeline(df3, **on)
    assert labels_of(r3, 2200, 2206) == {"post_reward"}
    assert labels_of(r3, 2400, 2406) == {"anticipatory"}


def test_licks_after_a_parked_spout_period_are_anticipatory():
    """Mirrors the real F19_F17 session structure.

    That recording opens with the spout parked out for about 31 s, retracts for about 9 s during
    which the mouse licks, then delivers a normal ~3 s reward. The parked stretch is not a
    reward, so nothing in the gap after it is reward clean-up and every lick there is
    anticipating the extension that follows.
    """
    fps = 30.0
    park = (int(1.0 * fps), int(30.8 * fps))          # ~30 s parked, an off-protocol placement
    rewards = [(int(39.9 * fps), int(42.9 * fps)),
               (int(65.8 * fps), int(68.7 * fps)),
               (int(95.0 * fps), int(98.0 * fps))]
    licks = [(int(t * fps), int(t * fps) + 3) for t in (31.5, 33.0, 34.2, 35.5, 36.4, 37.1,
                                                        38.0, 39.2)]
    df = make_df(int(120 * fps), extensions=[park] + rewards, tongue_runs=licks)
    r = run_pipeline(df, fps=fps)
    assert not r["accepted"][0], "the 30 s parked stretch must not count as a reward"
    assert all(r["accepted"][1:]), r["accepted"]
    got = [p.onset_frame for p in r["prots"] if p.label == "anticipatory"]
    assert len(got) == len(licks), (len(got), len(licks))
    assert all(p.trial_id == 2 for p in r["prots"] if p.label == "anticipatory")


def test_post_reward_licking_separated():
    """With the clean-up window on, licking immediately after the spout retracts is reward
    clean-up, but licking later in the same gap is anticipatory again."""
    df = make_df(4000, extensions=[(500, 680), (2000, 2180)],
                 tongue_runs=[(685, 700),      # right after retraction -> post_reward
                              (1200, 1204)])   # well into the gap -> anticipatory
    r = run_pipeline(df, post_reward_s=2.0)
    assert labels_of(r, 685, 700) == {"post_reward"}
    assert labels_of(r, 1200, 1204) == {"anticipatory"}
    # by default the window is off: any lick with the spout retracted is anticipatory
    r0 = run_pipeline(df)
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


def test_frame_params_follow_fps():
    """Frame-count settings are durations: the original values at 60 fps, halved at 30 fps."""
    at60 = resolve_frame_params(dict(DEFAULTS, fps=60.0))
    assert {k: at60[k] for k in FRAME_PARAMS} == dict(
        min_protrusion_frames=2, bout_gap_frames=20, min_extended_run=5,
        extension_merge_gap=30, lick_episode_gap=120, anticipatory_max_gap=120)
    # the defaults must agree with their own durations at the default frame rate (30 fps)
    assert DEFAULTS["fps"] == 30.0
    assert {k: resolve_frame_params(DEFAULTS)[k] for k in FRAME_PARAMS} == {k: DEFAULTS[k] for k in FRAME_PARAMS}
    at30 = resolve_frame_params(dict(DEFAULTS, fps=30.0))
    assert {k: at30[k] for k in FRAME_PARAMS} == dict(
        min_protrusion_frames=1, bout_gap_frames=10, min_extended_run=3,
        extension_merge_gap=15, lick_episode_gap=60, anticipatory_max_gap=60)
    pinned = resolve_frame_params(dict(DEFAULTS, fps=30.0), explicit={"min_protrusion_frames": 2})
    assert pinned["min_protrusion_frames"] == 2 and pinned["bout_gap_frames"] == 10


def test_single_frame_lick_kept_at_30fps():
    """At 30 fps a lick can show the tongue for a single frame; it must survive the blip filter."""
    df = make_df(3000, extensions=[(1000, 1090)], tongue_runs=[(980, 981), (1030, 1031)])
    at60 = run_pipeline(df)  # 60 fps defaults: a run must last 2 frames, so these are dropped
    assert not at60["valid"][980] and not at60["valid"][1030]
    cfg30 = resolve_frame_params(dict(DEFAULTS, fps=30.0))
    at30 = run_pipeline(df, fps=30.0, **{k: cfg30[k] for k in FRAME_PARAMS})
    assert at30["valid"][980] and at30["valid"][1030]


def test_onset_at_spout_motion():
    """A trial starts when the spout leaves rest, so licks at the incoming spout are not anticipatory."""
    df = make_df(3000, extensions=[(1000, 1180)], tongue_runs=[(900, 905), (994, 998)])
    for i, f in enumerate(range(990, 1000)):  # the spout travels out over frames 990-999
        df.loc[f, "spout_tip.x"] = RETRACTED[0] + (i + 1) / 11 * (EXTENDED[0] - RETRACTED[0])
        df.loc[f, "spout_tip.y"] = RETRACTED[1]
    a = run_pipeline(df, extension_onset="arrival")
    m = run_pipeline(df, extension_onset="motion")
    assert a["trials"][0][0] == 1000 and m["trials"][0][0] == 990, (a["trials"], m["trials"])
    assert labels_of(a, 994, 998) == {"anticipatory"}   # old behaviour: licks at the moving spout
    assert labels_of(m, 994, 998) == {"consummatory"}
    assert labels_of(m, 900, 905) == {"anticipatory"}   # licking before the spout moves is unaffected


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


def test_quality_report_flags_bad_tracking_and_stays_quiet_on_good():
    """The script has to say when the tracking is too poor to judge the model by.

    A model run on footage unlike its training set can miss the spout entirely or hold a
    "tongue" on a bright patch for many seconds. Both would otherwise produce a confident zero.
    """
    from detect_anticipatory_licks import report_input_quality
    cfg = resolve_frame_params(dict(DEFAULTS, fps=60.0))  # 60 fps synthetic data

    good = make_df(20000, extensions=[(s, s + 180) for s in (2000, 8000, 14000)],
                   tongue_runs=[(1900, 1906), (1920, 1926), (2100, 2130)])
    assert report_input_quality(good, cfg) == [], report_input_quality(good, cfg)

    # spout never tracked -> no trial can ever be found, and that must be said out loud
    no_spout = good.copy()
    no_spout["spout_tip.score"] = 0.0
    no_spout["spout_tip.x"] = np.nan
    assert any("spout_tip" in w for w in report_input_quality(no_spout, cfg))

    # a tongue stuck on for 30s, and on most of the session
    stuck = good.copy()
    stuck.loc[5000:23000, "tongue.score"] = 0.9
    stuck.loc[5000:23000, "tongue.x"] = 150.0
    stuck.loc[5000:23000, "tongue.y"] = 90.0
    w = report_input_quality(stuck, cfg)
    assert any("unbroken tongue" in x for x in w), w
    assert any("far more than a mouse actually licks" in x for x in w), w

    # no tongue at all
    none_df = good.copy()
    none_df["tongue.score"] = 0.0
    none_df["tongue.x"] = np.nan
    assert any("no tongue was detected" in x for x in report_input_quality(none_df, cfg))

    # an undersampled recording cannot resolve licks whatever the model does
    assert any("fps" in x for x in report_input_quality(good, {**cfg, "fps": 15.0}))


def test_csv_is_the_only_required_input():
    """The analysis CSV must be enough on its own.

    The script sits in the pipeline purely to turn the model's tongue and spout output into
    scored licks, so it must never reach for the video, the .slp, a rig log, or anything else.
    This runs it in an empty directory holding nothing but the script and one CSV.
    """
    import shutil, subprocess, sys, tempfile
    src = [f for f in glob.glob("*analysis*.csv") + glob.glob("*/*analysis*.csv")
           if not is_output_file(f)]
    if not src:
        print("    (SKIPPED, no session CSV present)")
        return
    script = "detect_anticipatory_licks.py"
    tmp = tempfile.mkdtemp(prefix="lickonly_")
    try:
        shutil.copy(script, tmp)
        shutil.copy(sorted(src)[0], os.path.join(tmp, "only.analysis.csv"))
        assert sorted(os.listdir(tmp)) == [script, "only.analysis.csv"]
        r = subprocess.run([sys.executable, script, "only.analysis.csv", "--fps", "60"],
                           cwd=tmp, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        made = set(os.listdir(tmp)) - {script, "only.analysis.csv"}
        for want in ("only.analysis_anticipatory_licks.csv", "only.analysis_trials.csv"):
            assert want in made, (want, made)
        # the module must not pull in anything beyond pandas/numpy from the stdlib set
        text = open(script).read()
        for banned in ("import cv2", "import matplotlib", "import sklearn", "import scipy"):
            assert banned not in text, banned
        print(f"    ran standalone in an empty dir, wrote {len(made)} files")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_bad_csv_reports_failure_in_exit_code():
    """A CSV the script cannot use must fail loudly rather than exiting clean."""
    import subprocess, sys, tempfile, shutil
    tmp = tempfile.mkdtemp(prefix="lickbad_")
    try:
        shutil.copy("detect_anticipatory_licks.py", tmp)
        pd.DataFrame({"frame_idx": [0, 1], "nose.x": [1.0, 1.0]}).to_csv(
            os.path.join(tmp, "broken.analysis.csv"), index=False)
        r = subprocess.run([sys.executable, "detect_anticipatory_licks.py", "broken.analysis.csv"],
                           cwd=tmp, capture_output=True, text=True)
        assert r.returncode != 0, "a malformed CSV exited 0"
        assert "ERROR" in r.stdout + r.stderr
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_setup_position_not_mistaken_for_reward():
    """A rig can park the spout somewhere other than the reward position.

    Raw recordings often open with the spout held out by hand while the animal is mounted, and
    that placement can occupy far more frames than the real rewards. Picking the biggest
    cluster away from rest would lock onto it and miss every trial, so the reward position is
    the one the spout returns to repeatedly for a consistent length of time.
    """
    setup = [(600, 12000)]                                   # one 190s placement, huge
    rewards = [(s, s + 180) for s in (14000, 20000, 26000, 32000, 38000)]
    df = make_df(45000, extensions=setup + rewards)
    # park the setup block at its own position, distinct from the reward position
    for s, e in setup:
        df.loc[s:e - 1, "spout_tip.x"] = 120.0
        df.loc[s:e - 1, "spout_tip.y"] = 60.0
    r = run_pipeline(df)
    assert r["reg"].extended is not None
    assert abs(r["reg"].extended[0] - EXTENDED[0]) < 5, r["reg"].extended
    assert abs(r["reg"].extended[1] - EXTENDED[1]) < 5, r["reg"].extended
    assert [tuple(t) for t in r["trials"]] == rewards, r["trials"]
    assert all(r["accepted"]), r["accepted"]


def test_setup_licking_excluded_from_session():
    """Licking while the spout is parked at a setup position is not anticipation, because the
    spout is sitting right there. The session starts after that placement ends."""
    setup = [(600, 12000)]
    rewards = [(s, s + 180) for s in (14000, 20000, 26000)]
    df = make_df(35000, extensions=setup + rewards,
                 tongue_runs=[(4000, 4010),      # during the setup placement
                              (16000, 16010)])   # a genuine between-trial lick
    for s, e in setup:
        df.loc[s:e - 1, "spout_tip.x"] = 120.0
        df.loc[s:e - 1, "spout_tip.y"] = 60.0
    r = run_pipeline(df)
    assert labels_of(r, 4000, 4010) == {"other"}
    assert labels_of(r, 16000, 16010) == {"anticipatory"}


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
    files = [f for f in glob.glob("*analysis*.csv") + glob.glob("*/*analysis*.csv")
              if not is_output_file(f)]
    if not files:
        print("    (SKIPPED, no session CSV present)")
        return
    df = load_sleap_csv(sorted(files)[0])
    base = _fingerprint(run_pipeline(df))
    shear_a = [[1, 0.45, 0], [0.2, 1, 0]]
    shear_b = [[1, -0.6, 0], [0.35, 1, 0]]
    # Moving or re-aiming a camera is a similarity transform. It preserves every distance
    # RATIO, and since the script only ever compares distances to learned distances, the result
    # has to come back bit for bit identical.
    exact = {
        "translate": dict(tx=500, ty=-300),
        "rotate 37deg": dict(ang=37),
        "rotate 113deg": dict(ang=113),
        "scale 0.6": dict(sx=0.6, sy=0.6),
        "scale 2.5": dict(sx=2.5, sy=2.5),
        "mirror": dict(mirror=True),
        "rot+scale+translate": dict(ang=25, sx=1.8, sy=1.8, tx=-120, ty=640),
    }
    # Shear, anisotropic scaling and perspective genuinely distort ratios, so a lick sitting
    # exactly on the learned ROI boundary can legitimately tip either way. Trials and the
    # overwhelming majority of licks must still survive unchanged.
    near = {
        "anisotropic": dict(sx=1.5, sy=0.8),
        "shear A": dict(M=shear_a),
        "shear B": dict(M=shear_b),
        "perspective": dict(persp=(3e-4, -2e-4)),
        "perspective strong": dict(persp=(8e-4, 5e-4)),
        "shear+perspective": dict(M=shear_a, persp=(4e-4, -3e-4)),
    }
    seps = []
    for name, kw in exact.items():
        got = _fingerprint(run_pipeline(warp(df, **kw)))
        assert got == base, f"{name} changed the result under a similarity transform"
    for name, kw in near.items():
        r = run_pipeline(warp(df, **kw))
        got = _fingerprint(r)
        assert len(got["accepted_trials"]) == len(base["accepted_trials"]), \
            f"{name}: trial count changed"
        shift = max(abs(x[0] - y[0]) for x, y in
                    zip(base["accepted_trials"], got["accepted_trials"]))
        assert shift <= 2, f"{name}: trial onset moved {shift} frames"
        diff = set(got["anti_onsets"]) ^ set(base["anti_onsets"])
        assert len(diff) <= max(2, 0.02 * len(base["anti_onsets"])), \
            f"{name}: {len(diff)} anticipatory licks differ"
        seps.append(r["reg"].separation)
    for kw in exact.values():
        seps.append(run_pipeline(warp(df, **kw))["reg"].separation)
    # the learned separation must actually track the zoom -- proving it is learned, not fixed
    assert max(seps) / min(seps) > 3, seps
    print(f"    {len(exact)} similarity poses identical, {len(near)} distorting poses within "
          f"tolerance; learned separation ranged {min(seps):.1f}-{max(seps):.1f}px")


# Pinned results per known SLEAP export of video 1. Keyed by the model-run stamp in the
# filename, so whichever CSV(s) are present get checked against the right expectation.
EXPECTED = {
    "260728_225759": dict(  # F19_F18, 960x600
        ret=(296.1, 336.0), ext=(352.2, 303.9), detected=15, accepted=14, leadin=12,
        window_median=195,
        counts={"anticipatory": 520, "consummatory": 801, "post_reward": 79, "other": 181},
        anti_prot=134, anti_lead=116, anti_solo=18),
    "260728_232648": dict(  # M13_M22, 800x500
        ret=(266.6, 299.9), ext=(300.0, 243.3), detected=18, accepted=13, leadin=4,
        window_median=26,
        counts={"anticipatory": 56, "consummatory": 785, "post_reward": 25, "other": 75},
        anti_prot=23, anti_lead=5, anti_solo=14),
    "260729_092644": dict(  # M15_F16 progressive ratio, 640x400, three spout positions
        ret=(138.7, 325.3), ext=(202.7, 269.2), detected=10, accepted=10, leadin=10,
        window_median=118,
        counts={"anticipatory": 2378, "consummatory": 1382, "post_reward": 57, "other": 1133},
        anti_prot=730, anti_lead=44, anti_solo=309),
}

# The same sessions with trials starting when the spout leaves rest (the default since
# extension_onset="motion"). Spout regions and trial counts are unchanged; onsets move 4-6 frames
# earlier, so licks at the incoming spout stop counting as anticipatory. EXPECTED above stays
# pinned to extension_onset="arrival", which reproduces the original behaviour exactly.
EXPECTED_MOTION = {
    "260728_225759": dict(detected=15, accepted=14, leadin=12, window_median=190,
        counts={"anticipatory": 490, "consummatory": 831, "post_reward": 79, "other": 181},
        anti_prot=129, anti_lead=111, anti_solo=18),
    "260728_232648": dict(detected=18, accepted=13, leadin=2, window_median=72,
        counts={"anticipatory": 51, "consummatory": 790, "post_reward": 25, "other": 75},
        anti_prot=20, anti_lead=2, anti_solo=14),
    "260729_092644": dict(detected=10, accepted=10, leadin=8, window_median=179,
        counts={"anticipatory": 2356, "consummatory": 1404, "post_reward": 57, "other": 1133},
        anti_prot=721, anti_lead=35, anti_solo=309),
}


def test_regression_real_csv():
    """Pin the measured numbers on every known real CSV present in the working directory."""
    files = [f for f in [f for f in glob.glob("*analysis*.csv") + glob.glob("*/*analysis*.csv")
              if not is_output_file(f)]
             if any(k in f for k in EXPECTED)]
    if not files:
        print(f"    (SKIPPED, no session CSV present -- looked for {len(EXPECTED)} known exports)")
        return
    print(f"    checking {len(files)} of {len(EXPECTED)} known sessions")
    for f, mode in [(f, m) for f in files for m in ("arrival", "motion")]:
        key = next(k for k in EXPECTED if k in f)
        exp = dict(EXPECTED[key], **(EXPECTED_MOTION[key] if mode == "motion" else {}))
        # pinned with the old 2 s clean-up window, which these counts were measured under
        r = run_pipeline(load_sleap_csv(f), extension_onset=mode, post_reward_s=2.0)
        reg = r["reg"]
        assert abs(reg.retracted[0] - exp["ret"][0]) < 1 and abs(reg.retracted[1] - exp["ret"][1]) < 1
        assert abs(reg.extended[0] - exp["ext"][0]) < 1 and abs(reg.extended[1] - exp["ext"][1]) < 1
        assert len(r["trials"]) == exp["detected"], (key, len(r["trials"]))
        assert sum(r["accepted"]) == exp["accepted"], (key, sum(r["accepted"]))
        # accepted trials are the rig's fixed-length reward deliveries, so they must all agree
        acc_dur = [(e - s) / 60 for (s, e), ok in zip(r["trials"], r["accepted"]) if ok]
        assert max(acc_dur) - min(acc_dur) < 0.25, (key, min(acc_dur), max(acc_dur))
        assert 2.5 < np.median(acc_dur) < 3.5, (key, np.median(acc_dur))
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
        print(f"    [{key} {mode}] {exp['detected']} detected / {exp['accepted']} accepted "
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
