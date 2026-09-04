"""Stage 2: turn a raw pose sequence into a list of swing events.

This is the heuristic MVP called for in the PRD's "Scope depth" decision:
no trained shot classifier exists yet (no labeled reference-swing dataset),
so swings are detected as speed peaks in the hitting wrist's motion rather
than recognized by a model. It's honest about what it is: a kinematic
signal, not a padel-shot classifier.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pose_extraction import PoseSequence, torso_length

WRIST_JOINTS = {"LEFT": "LEFT_WRIST", "RIGHT": "RIGHT_WRIST"}


@dataclass
class SwingEvent:
    index: int          # 1-based order in the clip
    frame_idx: int
    t: float             # seconds, peak contact instant
    peak_speed: float    # torso-lengths / second
    hand: str            # "LEFT" or "RIGHT"


def detect_dominant_hand(seq: PoseSequence) -> str:
    """Whichever wrist covers more total distance across the clip is taken
    as the hitting hand — the paddle arm moves far more than the off arm
    over a drill or rally."""
    totals = {"LEFT": 0.0, "RIGHT": 0.0}
    prev = {"LEFT": None, "RIGHT": None}
    for f in seq.frames:
        if not f.detected:
            continue
        for hand, joint in WRIST_JOINTS.items():
            if joint not in f.points:
                continue
            x, y, vis = f.points[joint]
            if vis < 0.3:
                continue
            if prev[hand] is not None:
                px, py = prev[hand]
                totals[hand] += float(np.hypot(x - px, y - py))
            prev[hand] = (x, y)
    return "RIGHT" if totals["RIGHT"] >= totals["LEFT"] else "LEFT"


def _wrist_speed_series(seq: PoseSequence, hand: str) -> np.ndarray:
    """Per-frame wrist speed in torso-lengths/second. NaN where undefined
    (missed detection, missing torso reference, or first frame)."""
    joint = WRIST_JOINTS[hand]
    speeds = np.full(len(seq.frames), np.nan)
    prev_xy = None
    prev_t = None
    for i, f in enumerate(seq.frames):
        if not f.detected or joint not in f.points:
            prev_xy, prev_t = None, None
            continue
        x, y, vis = f.points[joint]
        tl = torso_length(f.points)
        if vis < 0.3 or tl is None:
            prev_xy, prev_t = None, None
            continue
        if prev_xy is not None:
            dt = f.t - prev_t
            if dt > 0:
                dist = float(np.hypot(x - prev_xy[0], y - prev_xy[1]))
                speeds[i] = (dist / tl) / dt
        prev_xy, prev_t = (x, y), f.t

    # Light smoothing (3-frame moving average) to damp landmark jitter
    # without erasing a genuine swing peak, which spans several frames.
    smoothed = speeds.copy()
    for i in range(1, len(speeds) - 1):
        window = speeds[i - 1:i + 2]
        if not np.isnan(window).all():
            smoothed[i] = np.nanmean(window)
    return smoothed


def find_swings(
    seq: PoseSequence,
    hand: str | None = None,
    std_threshold: float = 1.2,
    min_gap_s: float = 0.5,
) -> tuple[str, np.ndarray, list[SwingEvent]]:
    """Detect swing events as prominent local peaks in wrist speed.

    Returns (hand_used, speed_series, events). speed_series is exposed so
    the report stage can plot/inspect it, not just consume the peaks.
    """
    hand = hand or detect_dominant_hand(seq)
    speeds = _wrist_speed_series(seq, hand)

    valid = speeds[~np.isnan(speeds)]
    if valid.size == 0:
        return hand, speeds, []

    threshold = float(np.mean(valid) + std_threshold * np.std(valid))

    # Candidate peaks: local maxima above threshold.
    candidates = []
    for i in range(1, len(speeds) - 1):
        v = speeds[i]
        if np.isnan(v) or v < threshold:
            continue
        left = speeds[i - 1] if not np.isnan(speeds[i - 1]) else -np.inf
        right = speeds[i + 1] if not np.isnan(speeds[i + 1]) else -np.inf
        if v >= left and v >= right:
            candidates.append(i)

    # Greedy non-max suppression by descending speed, enforcing min_gap_s.
    candidates.sort(key=lambda i: speeds[i], reverse=True)
    accepted: list[int] = []
    for i in candidates:
        t = seq.frames[i].t
        if all(abs(t - seq.frames[j].t) >= min_gap_s for j in accepted):
            accepted.append(i)
    accepted.sort()

    events = [
        SwingEvent(
            index=n + 1,
            frame_idx=i,
            t=seq.frames[i].t,
            peak_speed=float(speeds[i]),
            hand=hand,
        )
        for n, i in enumerate(accepted)
    ]
    return hand, speeds, events
