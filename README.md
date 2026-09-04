# Rebote (MVP prototype)

A phone-video swing analyzer, per the [Rebote PRD](.). This is the **heuristic
MVP** scoped in this build session — not the full product. Read
[What this is / isn't](#what-this-is--isnt) before trusting the numbers.

## Setup

```bash
cd rebote
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins `mediapipe==0.10.14` deliberately — see the comment
at the top of that file. Newer mediapipe crashes on this environment.

## Usage

```bash
python analyze.py sample_videos/man-playing-tennis-877.mp4
```

Prints a per-swing table (timestamp, shot zone, peak speed, elbow angle,
split-step) and a coaching note to the terminal, and writes:

- `output/<clip>_report.json` — the same data, structured
- `output/<clip>_annotated.mp4` — the source clip with the tracked skeleton
  and swing markers burned in (skip with `--no-video`)

Useful flags:

```bash
python analyze.py path/to/clip.mp4 \
  --hand left               # force hitting-hand side instead of auto-detecting
  --std-threshold 1.0       # lower = more sensitive swing detection
  --min-gap 0.4             # minimum seconds between two counted swings
```

### Turning on real coaching feedback

Right now feedback comes from a template that's grounded in the real
computed numbers but isn't an LLM. Set an API key and it upgrades itself,
no code or flag changes needed:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python analyze.py sample_videos/man-playing-tennis-877.mp4
```

The feedback panel's title will read `(claude-api)` instead of `(mock)`
when it's live.

## Architecture

```
video file
   -> rebote_lib/pose_extraction.py   MediaPipe Pose, per-frame joint positions
   -> rebote_lib/swing_detection.py   wrist-speed peaks -> swing events
   -> rebote_lib/metrics.py           contact height, elbow angle, follow-through,
                                      split-step, per-swing "shot zone"
   -> rebote_lib/coach.py             structured metrics -> feedback text
                                      (mock template, or live Claude API)
   -> rebote_lib/report.py            console table, JSON export, annotated video
```

## What this is / isn't

This build intentionally skipped the PRD's hardest, most expensive
dependency — a trained shot classifier — because that needs a labeled
dataset of real padel swings we don't have yet (the PRD calls this out as
the top risk: "cold-start reference data"). Instead:

- **"Shot zone" is not a shot classifier.** It buckets contact height into
  four bands (overhead / shoulder / waist / below-waist) that roughly map to
  where a bandeja, víbora, chiquita, etc. actually get hit — it does not
  know what stroke was played, only how high the contact point was.
- **Zone boundaries and the split-step heuristic are unvalidated guesses.**
  They're defined in `metrics.py` with comments explaining the reasoning,
  and are the first thing to recalibrate once real coach-reviewed footage
  is available.
- **The test clip is tennis, not padel** — the best free-license clip
  available with a clean single-player swing (see conversation for what
  was searched). Padel's overhead game (bandeja/víbora/remate) and the
  glass-wall rebound game aren't really exercised by it. Swap in a real
  padel clip any time by pointing `analyze.py` at it — nothing is
  tennis-specific in the code.
- **Single 2D camera, single player.** No court-position tracking, no
  partner-coverage analysis — that's the PRD's V2 scope (needs a wide,
  elevated shot of the full court, a different capture mode entirely).

## Known limitations worth knowing before you rely on this

- Contact-height and elbow-angle values come from 2D image-plane pose
  estimation — a camera angle that isn't roughly side-on to the swing will
  skew them (foreshortening).
- Auto hand-detection picks whichever wrist moved more over the whole
  clip; a clip with a lot of non-hitting arm movement (e.g. balancing,
  waving) could pick the wrong side. Override with `--hand`.
- Swing detection is a speed-threshold heuristic, not shot recognition — a
  fast non-swing arm motion (adjusting a cap, wiping sweat) can register
  as a false positive swing.
