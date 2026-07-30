# Validation of automated anticipatory-lick detection

Session F19_F17, recorded 2025-09-26. Video
`O_F19_F17_FP-250926-170646_F19_F17-250926-173045_Cam1_cropped.mp4`,
30 fps, 123,629 frames, 68.7 min. SLEAP export `260729_184036`.
Rig log `2025_09_26_OF19.csv`.

## What was compared

Manual scoring by eye against the automated pipeline, which is the SLEAP model plus
`detect_anticipatory_licks.py` reading the model's output CSV.

## Method, one rule used throughout

A **window** is 90 frames, 3.0 seconds, ending at the moment the spout reaches the mouse.
Spout arrival times come from the rig log, so they are objective. There are 16 rewards in the
session and therefore 16 possible windows.

**Window selection.** None. All **16** windows were scored, so this is a census and there is no
sampling question and no selection rule to defend. Total coverage 1,440 frames, 48.0 seconds.

**Manual scoring.** Every frame of every window was examined individually. A lick is one unbroken
tongue protrusion, recorded at its first visible frame, visible as a bright rounded or elongated
object emerging from the mouth into the region below the chin, which is dark when empty
(max pixel value 82 of 255). Contiguous bright frames are one lick. Paw or forelimb at the mouth,
faint grey brightening of the chin with no distinct protrusion, and the spout rod were all
excluded; the paw is distinguishable because it is larger, irregular, often shows separated
digits, and enters from below or the right rather than from the mouth. Scoring was done without
reference to the pipeline's output.

**Matching.** A manual lick and an automated lick count as the same event when their onsets fall
within 5 frames of each other. Greedy nearest-first, one-to-one, no reuse.

Nothing else was varied. Same window length, same crop, same scoring criterion, same matching
tolerance, in all sixteen windows.

## Results

| window | time | manual | automated | agreed | automated missed | automated invented |
|---|---|---|---|---|---|---|
| trial 1 | 37 s | 13 | 6 | 6 | 7 | 0 |
| trial 2 | 63 s | 11 | 12 | 11 | 0 | 1 |
| trial 3 | 112 s | 11 | 9 | 9 | 2 | 0 |
| trial 4 | 146 s | 10 | 7 | 7 | 3 | 0 |
| trial 5 | 174 s | 13 | 13 | 13 | 0 | 0 |
| trial 6 | 227 s | 14 | 13 | 13 | 1 | 0 |
| trial 7 | 388 s | 6 | 3 | 3 | 3 | 0 |
| trial 8 | 441 s | 10 | 7 | 7 | 3 | 0 |
| trial 9 | 834 s | 10 | 8 | 8 | 2 | 0 |
| trial 10 | 958 s | 4 | 5 | 4 | 0 | 1 |
| trial 11 | 1211 s | 6 | 7 | 6 | 0 | 1 |
| trial 12 | 1614 s | 9 | 6 | 6 | 3 | 0 |
| trial 13 | 1839 s | 4 | 6 | 4 | 0 | 2 |
| trial 14 | 1989 s | 6 | 4 | 4 | 2 | 0 |
| trial 15 | 2560 s | 3 | 2 | 2 | 1 | 0 |
| trial 16 | 3024 s | 10 | 9 | 9 | 1 | 0 |
| **total** | | **140** | **117** | **112** | **28** | **5** |

**Recall 80.0 percent**, 95 percent interval 72.6 to 85.8 percent.
The pipeline found 112 of the 140 licks that were actually present.

**Precision 95.7 percent**, 95 percent interval 90.4 to 98.2 percent.
112 of the 117 licks it reported were real; 5 were not.

Misses occur in 11 of the 16 windows and span the whole session, from 37 s to 3024 s, so the
shortfall is steady rather than confined to one part of the recording.

### What the pipeline misses

Misses concentrate in licks that follow closely on the previous one, which is what merging at
30 fps predicts:

| interval since previous manual lick | recall |
|---|---|
| 3 frames or less | 7/14 = 0.50 |
| 4 to 5 frames | 51/61 = 0.84 |
| 6 frames or more | 43/49 = 0.88 |
| first lick of the window | 11/16 = 0.69 |

The rate penalty is real but accounts for only part of the shortfall, because closely spaced licks
are just 14 of the 140. The first lick of a window is also missed disproportionately, 5 of 16,
which merging does not explain.

### What the pipeline invents

All 5 false positives were re-examined frame by frame at high magnification and all 5 are
genuinely not licks:

* **frames 1946, 28804, 55250** — faint grey brightening at the chin edge with no distinct
  protrusion. This was already the known failure mode and remains the commonest one.
* **frame 36386** — the forelimb held up at the mouth, with separated digits visible in the wider
  view. This is the paw confusion, and it is the first confirmed instance of the pipeline falling
  for it.
* **frame 55269** — a window-boundary artefact rather than a phantom lick. The tongue does contact
  the spout, but the contact starts at frame 55270, which is the arrival frame and therefore
  outside the window by definition. The detection is one frame early.

## Interpretation

The pipeline under-reports anticipatory licks by about a fifth and over-reports slightly. Its
whole-session figure of **218** anticipatory licks should be read as an undercount. Correcting for
both rates, 218 × 0.957 ÷ 0.800, puts the likely true number near **261**.

Separately, the pipeline recovered all 16 spout extensions in the session, matched one to one
against the rig log, with onset error under one frame after correcting a 644 ppm video clock
drift. Trial detection is therefore not a source of the lick shortfall.

## Limits of this validation

1. **Anticipation windows only.** All sixteen windows sit in the 3 seconds before a reward. The
   numbers describe licking that leads into a reward. Of the 218 licks the pipeline reports for
   the whole session, exactly 117 fall in these windows and the other 101 fall elsewhere. The
   licks elsewhere were not covered by this method, so recall and precision for them are
   unmeasured, and the whole-session estimate of 261 assumes the rates carry over.
2. **Two scoring batches, not perfectly consistent.** See below. The headline recall of 80.0
   percent blends two batches that differ, and is probably a slight underestimate.
3. **One session, one animal.** No replication across mice or days.
4. **Undercount cause not established.** The evidence above is consistent with 30 fps being
   marginal for a roughly 7 Hz behaviour, since a lick keeps the tongue visible for only 2 to 6
   frames, but the frame rate was not manipulated to test this directly.

### The batch inconsistency, stated plainly

The windows were scored in two sittings, trials 1, 4, 8, 12, 16 first and the remaining eleven
afterwards. The two batches disagree on how to split fast bouts:

| batch | manual | recall | inter-lick intervals of 3 frames or less |
|---|---|---|---|
| trials 1, 4, 8, 12, 16 | 52 | 35/52 = 0.673 | 12 of 43 = 28 percent |
| the other 11 | 88 | 77/88 = 0.875 | 2 of 67 = 3 percent |

The first batch contains nine inter-lick intervals of exactly 2 frames, which implies a 15 Hz
lick rate; the second contains none, and its modal interval is 4 frames, or 7.5 Hz. Since 15 Hz
is roughly double the rhythm seen anywhere else in this session's own data, the first batch
probably split some single protrusions in fast bouts into two, which inflates its manual counts
and depresses its recall. Trial 1 is the clearest case and was flagged as uncertain when it was
scored.

The census figures above are reported as scored, with no retroactive adjustment. If the first
batch were rescored under the second batch's convention, recall would rise from 80.0 percent
toward the second batch's 87.5 percent. Treat 80.0 percent as a conservative floor.

## Supporting files

* `handscore/handscored_F19_F17.csv` — the census: per window, manual and automated onset frames,
  agreements, misses and false positives
* `handscore/automated_anticipatory_licks_FULL.csv` — all 218 automated detections with frame,
  time, duration and confidence
* `handscore/_match.py` — the matching rule and the statistics, rerunnable
* `handscore/_zoom.py` — regenerates any frame range at the fixed crop and scale for re-checking
* `handscore/rig_log_decoding.md` — how the rig serial codes were identified
* `handscore/POSTER_DATA.md` — full-session comparisons against the rig log

## Manual scores used above, onset frames

* trial 1: 1108, 1111, 1169, 1171, 1176, 1180, 1182, 1184, 1187, 1189, 1191, 1193, 1195
* trial 2: 1896, 1902, 1914, 1927, 1932, 1937, 1942, 1951, 1958, 1966, 1972
* trial 3: 3355, 3365, 3381, 3395, 3414, 3420, 3424, 3428, 3432, 3436, 3440
* trial 4: 4432, 4445, 4450, 4454, 4458, 4462, 4468, 4473, 4477, 4483
* trial 5: 5246, 5255, 5261, 5265, 5270, 5276, 5282, 5286, 5292, 5297, 5301, 5305, 5311
* trial 6: 6812, 6816, 6820, 6824, 6828, 6832, 6836, 6842, 6847, 6851, 6856, 6861, 6875, 6887
* trial 7: 11704, 11712, 11716, 11720, 11724, 11727
* trial 8: 13233, 13239, 13248, 13288, 13296, 13300, 13304, 13309, 13313, 13319
* trial 9: 25007, 25053, 25061, 25065, 25069, 25073, 25077, 25081, 25085, 25090
* trial 10: 28823, 28829, 28835, 28841
* trial 11: 36373, 36400, 36405, 36410, 36413, 36419
* trial 12: 48476, 48485, 48489, 48494, 48498, 48502, 48504, 48510, 48515
* trial 13: 55245, 55254, 55258, 55264
* trial 14: 59741, 59746, 59750, 59756, 59761, 59768
* trial 15: 76791, 76796, 76877
* trial 16: 90726, 90774, 90780, 90785, 90787, 90790, 90795, 90800, 90806, 90813

Two of these onsets, 3395 and 76791, were found on a second pass that closed gaps in the first
pass's frame coverage; both are unambiguous at magnification. One, 6887, was added after
unblinding, when the pipeline's detection at 6886 prompted a recheck: a bright rounded protrusion
is present at the mouth and the spout rod is clearly separate from it, so it meets the criterion.
It is flagged here because it is the only manual onset that was not scored blind.
