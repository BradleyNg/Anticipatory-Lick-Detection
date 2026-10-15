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

## What it needs

The SLEAP analysis CSV. That is the whole list. Not the video, not the .slp, not a rig log.

The job is deliberately narrow. Take the tongue and spout the model reports, throw out what is
not a real lick, and sort what is left by where it falls around reward delivery, so the model
can be judged on how well it picks up anticipatory licking. A test enforces this by running the
script in an empty directory containing nothing but the script and one CSV.

Python 3 with pandas and numpy, and nothing else.

```
pip install pandas numpy
```

The two checking tools further down are separate and optional. Only the contact sheet one wants
opencv-python, and only if you want to eyeball results against video.

## Running the detector

```
python detect_anticipatory_licks.py path/to/session.analysis.csv
```

The default frame rate is 30 fps, which is what the current rig records. For the 2025 recordings,
which are 60 fps, add `--fps 60`.

Point it at a folder instead and it works through every analysis CSV inside, then writes a
combined summary next to them. Files the script has already written get skipped, so running it
over the same folder twice is safe. A CSV it cannot use is reported and makes the run exit
nonzero rather than passing quietly.

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

## It tells you when the tracking is not good enough

A model asked to run on footage unlike what it was trained on can miss a body part entirely, or
lock onto a bright patch and call it a tongue for twenty seconds straight. Either way the honest
answer is that nothing can be concluded, which is a lot more useful than a confident zero.

So before anything else the script checks the tracking and says what is wrong with it. It flags a
spout that is barely tracked, since without the spout there is no way to know when reward was
available and no trial can be found at all. It flags a tongue reported on an implausible share of
the session, a tongue held unbroken for longer than any real bout of licking, a tongue never
found at all, and a frame rate too low to separate individual licks.

That last one is worth knowing about. Mice lick at roughly seven per second with the tongue out
for something like a tenth of a second. At sixty frames per second a lick covers about six
frames, which is plenty. At fifteen it covers about one and a half, and a whole lick cycle covers
about two, so individual licks stop being separable no matter how good the model gets. Anything
under about twenty frames per second is not worth scoring per lick.

## How it holds up on real sessions

Three sessions from different mice, recorded at three different resolutions (960x600, 800x500
and 640x400) and all at sixty frames per second, so run with `--fps 60`.

|  | F19_F18 | M13_M22 | M15_F16 |
|---|---|---|---|
| extensions found, then accepted | 15, 14 | 18, 13 | 10, 10 |
| accepted trial length | 2.98 to 3.07s | 2.97 to 3.03s | 3.00 to 3.03s |
| anticipatory licks | 129 | 20 | 721 |
| in a run leading into a reward | 111 | 2 | 35 |
| lone single licks between trials | 18 | 14 | 309 |

The check worth trusting is the spout itself. Across all three sessions it sits at the learned
reward position on **100 percent** of the frames inside an accepted trial, and back at rest on
95 to 99.8 percent of the frames outside one. Accepted trial lengths land between 2.97 and 3.07
seconds everywhere, which is the rig repeating one fixed action: the logs program a 3.00 s
extension.

These numbers start each trial when the spout leaves rest (see "When a trial starts" below). With
`--extension-onset arrival`, the original behaviour, the same sessions give 134, 23 and 730
anticipatory licks and trial lengths of 2.87 to 2.98 s, short by the time the spout takes to
travel.

## When a trial starts

A trial used to start when the spout reached the reward position. The rig starts moving it 150
to 200 ms earlier, though, and the mouse is often already licking the spout on its way in. Those
licks were counted as anticipatory even when the tongue touched the spout. On a 30 fps session
checked against the rig log, arrival-based onsets ran a steady 4 to 6 frames late, and 7 to 8
percent of the "anticipatory" licks either began after the logged extension or set off the touch
sensor.

So a trial now starts right after the last frame the spout was still at rest, which puts the
onset within about 2 frames of the logged extension and drops those impossible licks to 2 to 5
percent. The remaining gap is about what the motor takes to start moving after the command. The
rest position and its radius are learned from the video, so this stays camera-independent.
`--extension-onset arrival` restores the original behaviour.

## Frame rate

The frame-count settings (how long a tongue must be out to count, how close licks must be to
share a bout, how far back a lead-in can reach, and so on) are durations underneath, and they are
converted at whatever `--fps` is given. At 60 fps that reproduces the original values exactly. At
30 fps the shortest licks show the tongue for a single frame, and a fixed two-frame minimum threw
away close to half of them on one 30 fps session, against the rig's touch sensor. The default
is 30 fps; pass `--fps` for any recording at another rate. Any frame-count option given explicitly on the
command line still wins.

The three sessions behave very differently, and that is real rather than a fault. M13 hardly
licks at all except when reward is there. M15 is a progressive ratio session, so its rewards
come further and further apart, at 53, 77, 158, 215, 269, 329, 527, 492 and 541 seconds, while
the mouse goes on licking the empty spout position throughout. That is where its much larger
anticipatory count comes from.

## Tests

```
python test_lick_logic.py
```

Twenty seven tests. Most of them build small fake sessions where the right answer is known and
check the labelling exactly, covering licking that crosses an extension, licking left over from
the previous trial, extensions broken into fragments by tracking dropout, sessions with no
rewards in them at all, lone single licks scattered between trials, a setup placement that
outweighs the real trials, a spout that takes several frames to travel out, and single-frame
licks at 30 fps. Two more run the script in an empty directory to prove the CSV is all it needs
and that a bad CSV fails loudly. The rest cover camera pose, the frame-rate conversion, and pin
the numbers from all three real sessions under both trial-onset rules.

## Data files

Recordings, tracking exports and anything the scripts write are kept out of the repo by
.gitignore. They are large and they are not mine to publish. The tests that need a session CSV
say so loudly and skip when it is absent, rather than quietly reporting a pass, so the rest of
the suite still runs on a fresh clone.

## Rewards close together

Rewards sometimes arrive close together, the spout retracting and going straight back out a few
seconds later. Licking in a gap like that is anticipating the reward about to arrive, not
finishing the one just given, so the clean-up window must not be allowed to eat it. Two things
prevent that. Licking that runs into the next extension is flagged as a lead-in and outranks the
clean-up carve, and the carve is separately capped at `--post-reward-max-frac` of the distance to
the next reward, which covers the case where the lead-in reach is set shorter than the clean-up
window.

Worth knowing that a long parked stretch at the head of a recording is not a reward, so the gap
after it is not clean-up either and licking there counts as anticipatory in full.

## Tuning

Set `--fps` to the recording's frame rate (30 is the default); everything counted in frames
follows it. The settings worth knowing about are `--post-reward-s`,
which is how long after a retraction still counts as finishing the reward rather than
anticipating the next one, and `--tongue-score-min`, which is the SLEAP confidence needed before
a tongue is believed. Setting `--post-reward-s 0` folds that licking into the anticipatory pile
instead. There is also `--lick-episode-gap`, which only decides where a lead-in run is judged to
begin and no longer affects which licks are counted. Run any of the scripts with `--help` for
the rest.
