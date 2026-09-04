"""Stage 1 of the Rebote pipeline: video -> per-frame pose landmarks.

Uses MediaPipe's legacy `solutions.pose` API (see requirements.txt for why
that specific version is pinned). Landmarks are stored in pixel coordinates
(not MediaPipe's normalized 0-1 space) so downstream distance/speed math
doesn't need to know frame size.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

# The joints the metrics stage actually needs. MediaPipe's Pose model
# produces 33 landmarks total; we only carry the ones swing/metric logic
# touches, keyed by the same names as mp_pose.PoseLandmark for clarity.
JOINTS = (
    "NOSE",
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE",
)


@dataclass
class FrameLandmarks:
    idx: int
    t: float  # seconds from clip start
    points: dict  # joint name -> (x_px, y_px, visibility)
    detected: bool


@dataclass
class PoseSequence:
    fps: float
    width: int
    height: int
    frames: list = field(default_factory=list)  # list[FrameLandmarks]

    @property
    def detection_rate(self) -> float:
        if not self.frames:
            return 0.0
        return sum(f.detected for f in self.frames) / len(self.frames)


def extract_pose_sequence(video_path: str, model_complexity: int = 1) -> PoseSequence:
    """Run pose estimation over every frame of `video_path`.

    model_complexity: 0 (fastest/least accurate) - 2 (slowest/most accurate),
    matching MediaPipe Pose's own scale.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    seq = PoseSequence(fps=fps, width=width, height=height)

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            points = {}
            detected = bool(result.pose_landmarks)
            if detected:
                lm = result.pose_landmarks.landmark
                for name in JOINTS:
                    p = lm[mp_pose.PoseLandmark[name].value]
                    points[name] = (p.x * width, p.y * height, p.visibility)

            seq.frames.append(
                FrameLandmarks(idx=idx, t=idx / fps, points=points, detected=detected)
            )
            idx += 1

    cap.release()
    return seq


def torso_length(points: dict) -> float | None:
    """Shoulder-midpoint to hip-midpoint distance in pixels, used everywhere
    downstream to normalize speeds/heights so results don't depend on how
    far the camera was from the player."""
    needed = ("LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP")
    if not all(k in points for k in needed):
        return None
    sx = (points["LEFT_SHOULDER"][0] + points["RIGHT_SHOULDER"][0]) / 2
    sy = (points["LEFT_SHOULDER"][1] + points["RIGHT_SHOULDER"][1]) / 2
    hx = (points["LEFT_HIP"][0] + points["RIGHT_HIP"][0]) / 2
    hy = (points["LEFT_HIP"][1] + points["RIGHT_HIP"][1]) / 2
    d = float(np.hypot(sx - hx, sy - hy))
    return d if d > 1e-3 else None
