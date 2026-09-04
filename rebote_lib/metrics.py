"""Stage 3: per-swing biomechanical metrics from pose + swing events.

These are the heuristic stand-ins for the PRD's "technique scoring" module.
Thresholds below (zone boundaries, split-step amplitude) are reasonable
starting guesses, not calibrated against real coach-labeled data — that
calibration is exactly the "cold-start reference data" risk the PRD flags.
Treat the numbers as directionally useful, not clinically precise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pose_extraction import PoseSequence, torso_length
from .swing_detection import SwingEvent

ELBOW_JOINTS = {"LEFT": ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
                "RIGHT": ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST")}
WRIST_JOINTS = {"LEFT": "LEFT_WRIST", "RIGHT": "RIGHT_WRIST"}

# contact_height_ratio boundaries -> shot zone label. Positive = above
# shoulder line, negative = below, in units of torso length.
ZONES = (
    (0.15, "Overhead (remate / víbora zone)"),
    (-0.05, "Shoulder-height (bandeja / volea zone)"),
    (-0.45, "Waist-level (chiquita / drive zone)"),
    (-np.inf, "Below-waist (defensive dig zone)"),
)


@dataclass
class SwingMetrics:
    event: SwingEvent
    contact_height_ratio: float | None
    elbow_angle_deg: float | None
    follow_through_delta: float | None
    split_step_detected: bool
    zone: str


def _angle_deg(a, b, c) -> float | None:
    """Angle at vertex b, given three (x, y) points."""
    v1 = np.array([a[0] - b[0], a[1] - b[1]])
    v2 = np.array([c[0] - b[0], c[1] - b[1]])
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def _zone_for(ratio: float | None) -> str:
    if ratio is None:
        return "Unknown"
    for lower, label in ZONES:
        if ratio >= lower:
            return label
    return ZONES[-1][1]


def _split_step_detected(
    seq: PoseSequence, peak_idx: int, window_s: float = 0.6, min_amplitude: float = 0.05
) -> bool:
    """Proxy for a pre-shot split-step: a brief knee-bend (hips drop toward
    ankles, then recover) in the window just before contact."""
    fps = seq.fps
    start = max(0, peak_idx - int(window_s * fps))
    series = []
    for f in seq.frames[start:peak_idx]:
        if not f.detected:
            continue
        pts = f.points
        if not all(k in pts for k in ("LEFT_HIP", "RIGHT_HIP", "LEFT_ANKLE", "RIGHT_ANKLE")):
            continue
        tl = torso_length(pts)
        if tl is None:
            continue
        hip_y = (pts["LEFT_HIP"][1] + pts["RIGHT_HIP"][1]) / 2
        ankle_y = (pts["LEFT_ANKLE"][1] + pts["RIGHT_ANKLE"][1]) / 2
        series.append((ankle_y - hip_y) / tl)  # leg extension, shrinks on crouch

    if len(series) < 4:
        return False
    arr = np.array(series)
    dip = float(np.max(arr) - np.min(arr))
    min_pos = int(np.argmin(arr))
    recovers = min_pos < len(arr) - 1  # crouch is followed by standing back up
    return dip >= min_amplitude and recovers


def compute_swing_metrics(seq: PoseSequence, event: SwingEvent) -> SwingMetrics:
    frame = seq.frames[event.frame_idx]
    pts = frame.points
    tl = torso_length(pts)

    shoulder_joint = f"{event.hand}_SHOULDER"
    elbow_joint = f"{event.hand}_ELBOW"
    wrist_joint = WRIST_JOINTS[event.hand]

    contact_height_ratio = None
    if tl and shoulder_joint in pts and wrist_joint in pts:
        shoulder_y = pts[shoulder_joint][1]
        wrist_y = pts[wrist_joint][1]
        contact_height_ratio = (shoulder_y - wrist_y) / tl  # +up / -down

    elbow_angle_deg = None
    s, e, w = ELBOW_JOINTS[event.hand]
    if all(j in pts for j in (s, e, w)):
        elbow_angle_deg = _angle_deg(pts[s][:2], pts[e][:2], pts[w][:2])

    follow_through_delta = None
    follow_frame_idx = min(len(seq.frames) - 1, event.frame_idx + int(0.2 * seq.fps))
    if follow_frame_idx != event.frame_idx:
        fpts = seq.frames[follow_frame_idx].points
        ftl = torso_length(fpts)
        if ftl and shoulder_joint in fpts and wrist_joint in fpts and contact_height_ratio is not None:
            f_ratio = (fpts[shoulder_joint][1] - fpts[wrist_joint][1]) / ftl
            follow_through_delta = f_ratio - contact_height_ratio

    return SwingMetrics(
        event=event,
        contact_height_ratio=contact_height_ratio,
        elbow_angle_deg=elbow_angle_deg,
        follow_through_delta=follow_through_delta,
        split_step_detected=_split_step_detected(seq, event.frame_idx),
        zone=_zone_for(contact_height_ratio),
    )


def compute_all(seq: PoseSequence, events: list[SwingEvent]) -> list[SwingMetrics]:
    return [compute_swing_metrics(seq, e) for e in events]
