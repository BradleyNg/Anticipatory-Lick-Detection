# Anticipatory Lick Detection

Tools for pulling anticipatory licks out of head fixed oHrbets sessions, starting from SLEAP
tongue and spout tracking.

The rig already logs every time the tongue touches the spout, but it cannot see the licks that
never reach it, and those are the ones this is for. So the video has to do that work.

## How a lick gets classified

The spout only swings in toward the mouse once it has turned the wheel enough to earn a reward,
and it pulls back afterward. Every extension is therefore a reward trial, and licks are sorted
by where they land around one.

A lick counts as anticipatory whenever the spout is pulled back during the session, since no
reward is available at that moment. Licking while the spout is out is consummatory. Licking in
the couple of seconds right after the spout retracts is treated separately as post reward,
because the mouse is finishing what it just earned rather than anticipating anything.

Bout size is never used as a filter. The mouse often licks once between trials without working
up to a proper run, and a lone lick like that is still a lick made with no reward available, so
it is reported the same as any other. On the sessions I checked, most of the spontaneous
anticipatory licks are exactly these single licks, so dropping them would lose a real chunk of
the data.

Runs that build straight into a spout extension are still worth telling apart, so every lick
carries a `leads_into_reward` flag along with the trial it belongs to and how long it fell
before that trial's reward. Nothing gets excluded to make that distinction. It is a column, so
either view is one filter away.

## Finding the reward position

A recording usually holds more than two spout positions. Besides resting and delivering reward,
the spout gets parked or held by hand while the animal is being mounted, and that placement can
occupy far more frames than every real trial put together. One of the sessions here spends 175
seconds parked in a spot that is nowhere near the reward position, against ten rewards of three
seconds each.

So the reward position is not taken to be the busiest place the spout visits. It is the place
the spout keeps returning to for the **same length of time**, because that is what the rig
actually does. Candidate positions are scored on how many repeat visits agree on a duration, and
points strung along the travel path get rejected that way too, since their visit lengths
scatter.

## Camera angles

Every recording sits at a different angle and zoom, so nothing in the code refers to a fixed
pixel position. The spout positions are learned from whichever video is being processed, and
every threshold after that is either a distance to one of those learned positions or a ratio.
The one fixed number is the SLEAP confidence cutoff, which belongs to the model rather than to
the camera.

A test re-projects a real session through thirteen camera poses. Seven of them are similarity
transforms, meaning translation, rotation, zoom and mirroring, which is what moving a camera
actually does. Those preserve every distance ratio, so the test demands bit for bit identical
output. The other six add shear, anisotropic scaling and perspective, which genuinely distort
ratios, so there the test allows a lick sitting exactly on a learned boundary to tip either way
and checks that the trials and effectively all the licks still come back.

## Getting set up

Python 3 with pandas and numpy.

```
pip install pandas numpy
```

The contact sheet script also wants opencv-python, but only if you plan to check results against
video.

## Running the detector

```
python detect_anticipatory_licks.py path/to/video.analysis.csv
```

Point it at a folder instead and it works through every analysis CSV inside, then writes a
combined summary next to them.

Four files come out per video.

| file | what is in it |
|---|---|
| `<name>_anticipatory_licks.csv` | one row per anticipatory tongue protrusion, which is the main result |
| `<name>_anticipatory_bouts.csv` | one row per anticipatory licking period |
| `<name>_lick_bouts.csv` | every bout across all categories, with a trial number |
| `<name>_trials.csv` | one row per spout extension |

The columns worth knowing in the licks file are `leads_into_reward`, which is true when the lick
sits in a run building into an extension, `isolated`, which is true when it was the only lick in
its bout, and `s_to_extension`, which is how many seconds passed before that trial's reward
arrived. Filtering on those gets you established licking runs only, or lone licks only, without
having to rerun anything.

## Raw recordings

The videos are not trimmed, so the mouse is still being put on the rig at the start and is being
taken off at the end. Extensions near either edge are not comparable to the ones in the middle,
and the script drops them by itself. It compares how long each extension lasts against the
session median, works out roughly where the real session sits, and flags anything that falls
outside. Nothing disappears quietly. Every extension still turns up in the trials file with
columns saying whether it was kept and why.

On the session I tested this keeps fourteen real trials, all lasting almost exactly three
seconds, and throws out a thirty second stretch at the very start where the spout was parked out
before the rig had started, along with a short blip or two after the last real trial.

If the automatic call is ever wrong, you can set the bounds yourself with `--trim-start-s` and
`--trim-end-s`, gate on duration with `--min-extension-s` and `--max-extension-s`, or switch the
whole thing off with `--keep-all-trials`.

## Checking against hand scored licks

```
python compare_ground_truth.py hand_scored.csv --licks <name>_anticipatory_licks.csv --bouts <name>_lick_bouts.csv
```

It works out which column holds the timestamps, and whether they are in seconds, milliseconds or
frame numbers, by seeing which reading fits inside the length of the video. Files with one row
per lick and files with a start and end per period both work.

Alongside precision and recall it reports the median offset between matched pairs, which is worth
reading. A large offset usually means the two clocks are not lined up rather than the detection
being wrong, and `--offset-scan` will go looking for the shift.

The most useful part is what it says about the hand scored licks it could not match. It tells you
whether nothing was detected at that moment at all, which is a tracking miss, or whether
something was detected and filed under a different category, which is a classification problem.
Those two need different fixes.

## Checking against the video

```
python review_bouts.py path/to/video.mp4 --bouts <name>_lick_bouts.csv --sleap tracking.analysis.csv --class anticipatory
```

This pulls the best frame out of each bout, draws the tracked points and the two learned spout
positions onto it, and tiles them into a sheet so a whole session can be scanned without
scrubbing through video. Red is the tongue, cyan is the spout tip, yellow is the nose, and the
two hollow circles marked R and E are the retracted and extended positions. For an anticipatory
bout the spout should be sitting at R with a real tongue under the red dot.

Swapping in `--class other` shows what the trial filter threw out, which is a quick way to sanity
check the split.

## How it holds up on real sessions

Three sessions from different mice, recorded at three different resolutions (960x600, 800x500
and 640x400) and all at sixty frames per second.

|  | F19_F18 | M13_M22 | M15_F16 |
|---|---|---|---|
| extensions found, then accepted | 15, 14 | 18, 13 | 10, 10 |
| accepted trial length | 2.93 to 2.98s | 2.87 to 2.98s | 2.90 to 2.93s |
| anticipatory licks | 134 | 23 | 730 |
| in a run leading into a reward | 116 | 5 | 44 |
| lone single licks between trials | 18 | 14 | 309 |

The check worth trusting is the spout itself. Across all three sessions it sits at the learned
reward position on **100 percent** of the frames inside an accepted trial, and back at rest on
95 to 99.8 percent of the frames outside one. Accepted trial lengths land between 2.87 and 2.98
seconds everywhere, which is the rig repeating one fixed action.

The three sessions behave very differently, and that is real rather than a fault. M13 hardly
licks at all except when reward is there. M15 is a progressive ratio session, so its rewards
come further and further apart, at 53, 77, 158, 215, 269, 329, 527, 492 and 541 seconds, while
the mouse goes on licking the empty spout position throughout. That is where its much larger
anticipatory count comes from.

## Tests

```
python test_lick_logic.py
```

Nineteen tests. Most of them build small fake sessions where the right answer is known and check
the labelling exactly, covering licking that crosses an extension, licking left over from the
previous trial, extensions broken into fragments by tracking dropout, sessions with no rewards
in them at all, lone single licks scattered between trials, and a setup placement that outweighs
the real trials. The rest cover camera pose and pin the numbers from all three real sessions.

## Data files

Recordings, tracking exports and anything the scripts write are kept out of the repo by
.gitignore. They are large and they are not mine to publish. The tests that need a session CSV
say so loudly and skip when it is absent, rather than quietly reporting a pass, so the rest of
the suite still runs on a fresh clone.

## Tuning

Defaults assume sixty frames per second. The ones worth knowing about are `--post-reward-s`,
which is how long after a retraction still counts as finishing the reward rather than
anticipating the next one, and `--tongue-score-min`, which is the SLEAP confidence needed before
a tongue is believed. Setting `--post-reward-s 0` folds that licking into the anticipatory pile
instead. There is also `--lick-episode-gap`, which only decides where a lead-in run is judged to
begin and no longer affects which licks are counted. Run any of the scripts with `--help` for
the rest.
