# Handoff: finish manual scoring of anticipatory licks, session F19_F17

Read this before scoring anything. The value of this dataset is that every window is scored the
**same way**. If you change the window size, the crop, the criterion or the matching rule, the
already-scored windows stop being comparable and the whole thing has to be redone. Do not
improve the method. Just continue it.

## Goal

Score **all 16** reward-anticipation windows in one session, by eye, frame by frame, and compare
to what the automated pipeline reported in the same frames.

This is a **census, not a sample**. All 16 get scored, so there is no window-selection question.
**All 16 are now scored.** Results are in `VALIDATION_F19_F17.md`; the per-window data is in
`handscore/handscored_F19_F17.csv` and the statistics rerun with `python handscore/_match.py`.

## The video and the pipeline output

* video `C:\Users\1moon\Downloads\O_F19_F17_FP-250926-170646_F19_F17-250926-173045_Cam1_cropped.mp4`
  30 fps, 123,629 frames, 550x500
* pipeline output `f19f17/F19_F17.analysis_anticipatory_licks.csv`, 218 detections.
  Regenerate with
  `python detect_anticipatory_licks.py f19f17/F19_F17.analysis.csv --fps 30 --out-dir f19f17`
* rig log `C:\Users\1moon\Downloads\2025_09_26_OF19.csv`. Serial code 14 is the solenoid pulse,
  five per reward. The first pulse of each burst is when the spout reaches the mouse.
* video-to-rig alignment `video_seconds = 1.000644 * rig_seconds + 30.443`, which includes a
  644 ppm clock drift. Do not use a plain offset, it drifts to 2 s by the end of the session.

## Window definition, fixed

A window is the **90 frames (3.0 s) immediately before the spout arrives**, arrival frame
excluded. Arrival frames are already computed below, so nothing needs recalculating.

| trial | arrival frame | window frames | window time | status |
|---|---|---|---|---|
| 1 | 1198 | 1108-1197 | 36.93-39.93 s | DONE |
| 2 | 1976 | 1886-1975 | 62.87-65.87 s | DONE |
| 3 | 3443 | 3353-3442 | 111.77-114.77 s | DONE |
| 4 | 4484 | 4394-4483 | 146.47-149.47 s | DONE |
| 5 | 5314 | 5224-5313 | 174.13-177.13 s | DONE |
| 6 | 6889 | 6799-6888 | 226.63-229.63 s | DONE |
| 7 | 11730 | 11640-11729 | 388.00-391.00 s | DONE |
| 8 | 13322 | 13232-13321 | 441.07-444.07 s | DONE |
| 9 | 25095 | 25005-25094 | 833.50-836.50 s | DONE |
| 10 | 28842 | 28752-28841 | 958.40-961.40 s | DONE |
| 11 | 36423 | 36333-36422 | 1211.10-1214.10 s | DONE |
| 12 | 48520 | 48430-48519 | 1614.33-1617.33 s | DONE |
| 13 | 55270 | 55180-55269 | 1839.33-1842.33 s | DONE |
| 14 | 59769 | 59679-59768 | 1989.30-1992.30 s | DONE |
| 15 | 76879 | 76789-76878 | 2559.63-2562.63 s | DONE |
| 16 | 90814 | 90724-90813 | 3024.13-3027.13 s | DONE |

## Contact sheets

One sheet per window, 90 frames each, at `handscore/F*.png` (and `G*.png`, the same frames in a
7-column layout that avoids downscaling). Read them with the Read tool and score by eye, then
magnify anything ambiguous with `_zoom.py`.

This is the exact code that made them. Crop and scale must not change.

```python
import cv2, numpy as np
v=r'C:\Users\1moon\Downloads\O_F19_F17_FP-250926-170646_F19_F17-250926-173045_Cam1_cropped.mp4'
cap=cv2.VideoCapture(v)
arrive={2:1976,3:3443,5:5314,6:6889,7:11730,9:25095,10:28842,11:36423,13:55270,14:59769,15:76879}
for t,a in arrive.items():
    f0=a-90; cap.set(cv2.CAP_PROP_POS_FRAMES,f0); buf=[]
    for fr in range(f0,a):
        ok,img=cap.read()
        if not ok: break
        tl=cv2.resize(img[205:300,138:278],(224,152),interpolation=cv2.INTER_CUBIC)
        cv2.putText(tl,f"{fr}",(2,148),cv2.FONT_HERSHEY_SIMPLEX,0.36,(0,255,255),1)
        buf.append(tl)
    while len(buf)<90: buf.append(np.zeros_like(buf[0]))
    cv2.imwrite(f"handscore/F{t:02d}.png",np.vstack([np.hstack(buf[j:j+9]) for j in range(0,90,9)]))
cap.release()
```

## Scoring criterion, fixed

One **lick** is one unbroken tongue protrusion. Record the **first frame** it is visible.

A tongue is a **bright, smooth-edged, rounded or elongated blob emerging from the mouth
downward into the region below the chin.** That region is near black when empty, max pixel value
82 of 255, so a real tongue is high contrast and obvious.

Two frames of no tongue between bright frames means two separate licks. Contiguous bright frames
are one lick.

### Things that are not a tongue

* **Paw or forelimb.** Much larger, irregular outline, often with visible claw separations, and
  it enters from below or the right rather than from the mouth. This is the main confusion. Some
  windows have the paw up at the mouth for seconds at a time.
* **Faint grey brightening at the chin edge** with no distinct protrusion. Do not count it. Three
  of the pipeline's false positives were exactly this.
* **The spout rod.** A thin straight bright line entering from the lower left. When retracted its
  tip sits near (150, 290) in full-frame coordinates. It thickens and reaches the mouth near
  (215, 245) at arrival, which is the end of the window.

### Uncertainty to expect

In fast bouts, roughly 5 licks per second or more, licks blur together and separating them is
genuinely ambiguous. Trial 1 was like this and its count of 13 is less certain than the others.
Note it when a window is like this rather than pretending the count is exact.

## Matching rule, fixed

A manual lick and a pipeline lick are the same event if their onset frames are **within 5
frames** of each other. Greedy nearest-first, one-to-one, no reuse.

## Results, all 16 of 16

| batch | manual | automated | agreed | missed | invented |
|---|---|---|---|---|---|
| trials 1, 4, 8, 12, 16 (first sitting) | 52 | 35 | 35 | 17 | 0 |
| the other 11 (second sitting) | 88 | 82 | 77 | 11 | 5 |
| **total** | **140** | **117** | **112** | **28** | **5** |

recall 0.800 (95% 0.726-0.858), precision 0.957 (95% 0.904-0.982), 1440 frames = 48.0 s checked

Per-window numbers and all onset frames: `handscore/handscored_F19_F17.csv`.
Rerun the statistics with `python handscore/_match.py` (manual onsets are hard-coded in it).

## Open issue: the two batches are not perfectly consistent

The first sitting has nine inter-lick intervals of exactly 2 frames, implying 15 Hz. The second
has none; its modal interval is 4 frames, 7.5 Hz. The first sitting probably split some single
protrusions in fast bouts into two, which inflates its manual count and depresses its recall
(0.673 vs 0.875). Nothing has been changed retroactively. **If you want the batches consistent,
the thing to do is rescore trials 1, 4, 8, 12, 16 under the second convention** — a dark gap is
required between protrusions, one bright run is one lick. Trial 1 is the extreme case.

## Notes for anyone rechecking

* `handscore/_zoom.py f0 f1 cols out.png` regenerates any frame range at the fixed crop and scale.
  Fewer columns means more magnification. This is how the ambiguous frames were adjudicated.
* Wider crops (`img[180:330, 110:310]`) were used only to tell paw from tongue, never to score.
  The paw shows separated digits and connects to a limb entering from below or the right.
* Frames where the spout rod reaches the mouth at the very end of a window are the hardest calls.
  A tongue is a rounded blob that is clearly separate from the rod's straight bar. Confirmed
  tongues at arrival: 3440, 11727, 28841, 59768, 76877, and 6887. Confirmed rod-only: 6886.
* 3395 and 76791 were missed on the first pass because those frame bands were only ever viewed on
  the 90-frame contact sheet, never magnified. Cover every band at magnification.

## Done

`VALIDATION_F19_F17.md` has been updated to the full census: numbers replaced, the window-selection
section rewritten as a census, the "small sample" limitation dropped, the other three kept, and the
batch inconsistency added as a fourth. The 101 detections outside these windows remain unmeasured
and that is stated there.

## Do not

* change the window length, the crop, the criterion or the matching tolerance
* re-score windows already done, in order to "check", unless the user asks. The one exception
  worth doing is the batch-consistency rescore described above, and that is the user's call
* substitute an automated pixel detector for manual scoring. That was considered and rejected,
  because two unvalidated detectors agreeing proves nothing
* call the rig log a source of anticipatory ground truth. Its lick channel is code 31 and it only
  fires on tongue-to-spout contact, so anticipatory licks are invisible to it by construction
