"""Stage 5: render results - console report, JSON export, annotated video."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .coach import SessionSummary
from .metrics import SwingMetrics
from .pose_extraction import PoseSequence

SKELETON_CONNECTIONS = (
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER", "LEFT_ELBOW"), ("LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"), ("RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_SHOULDER", "LEFT_HIP"), ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_HIP", "RIGHT_HIP"),
    ("LEFT_HIP", "LEFT_KNEE"), ("LEFT_KNEE", "LEFT_ANKLE"),
    ("RIGHT_HIP", "RIGHT_KNEE"), ("RIGHT_KNEE", "RIGHT_ANKLE"),
)


def render_console(
    video_path: str,
    seq: PoseSequence,
    metrics: list[SwingMetrics],
    summary: SessionSummary,
    feedback: str,
    feedback_source: str,
) -> None:
    console = Console()
    console.rule(f"[bold]Rebote[/bold] · {Path(video_path).name}")
    console.print(
        f"{len(seq.frames)} frames · {seq.fps:.1f} fps · pose tracked on "
        f"{seq.detection_rate:.0%} of frames · hitting hand: {summary.hand.title()}"
    )

    if metrics:
        table = Table(show_header=True, header_style="bold")
        table.add_column("#", justify="right")
        table.add_column("Time")
        table.add_column("Zone")
        table.add_column("Peak speed", justify="right")
        table.add_column("Elbow °", justify="right")
        table.add_column("Split-step", justify="center")
        for m in metrics:
            mm, ss = divmod(m.event.t, 60)
            table.add_row(
                str(m.event.index),
                f"{int(mm):02d}:{ss:04.1f}",
                m.zone,
                f"{m.event.peak_speed:.1f}",
                f"{m.elbow_angle_deg:.0f}" if m.elbow_angle_deg is not None else "-",
                "✓" if m.split_step_detected else "·",
            )
        console.print(table)
    else:
        console.print("[yellow]No swings detected - try a different --hand, or check the clip "
                       "actually shows hitting action clearly.[/yellow]")

    console.print(Panel(feedback, title=f"Coaching feedback ({feedback_source})", border_style="green"))


def export_json(path: str, video_path: str, seq: PoseSequence, metrics: list[SwingMetrics],
                 summary: SessionSummary, feedback: str, feedback_source: str) -> None:
    data = {
        "video": str(video_path),
        "fps": seq.fps,
        "frame_count": len(seq.frames),
        "pose_detection_rate": seq.detection_rate,
        "hand": summary.hand,
        "swings": [
            {
                "index": m.event.index,
                "t": round(m.event.t, 3),
                "peak_speed_torso_lengths_per_s": round(m.event.peak_speed, 3),
                "contact_height_ratio": m.contact_height_ratio,
                "elbow_angle_deg": m.elbow_angle_deg,
                "follow_through_delta": m.follow_through_delta,
                "split_step_detected": m.split_step_detected,
                "zone": m.zone,
            }
            for m in metrics
        ],
        "summary": {
            "swing_count": summary.swing_count,
            "zone_counts": dict(summary.zone_counts),
            "avg_peak_speed": summary.avg_peak_speed,
            "split_step_rate": summary.split_step_rate,
        },
        "feedback": feedback,
        "feedback_source": feedback_source,
    }
    Path(path).write_text(json.dumps(data, indent=2))


def annotate_video(video_path: str, seq: PoseSequence, metrics: list[SwingMetrics], out_path: str) -> None:
    """Re-reads the source video and writes a copy with the tracked skeleton
    and swing markers burned in - visual proof the CV stage is actually
    seeing what it claims to see."""
    peak_frames = {m.event.frame_idx: m for m in metrics}
    # Show each swing label for a short window so it's readable, not a single flash-frame.
    label_window = max(1, int(0.4 * seq.fps))
    labeled_frames = {}
    for fidx, m in peak_frames.items():
        for off in range(-label_window // 2, label_window // 2 + 1):
            labeled_frames.setdefault(fidx + off, m)

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, seq.fps, (seq.width, seq.height))

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or idx >= len(seq.frames):
            break
        fl = seq.frames[idx]
        if fl.detected:
            pts = fl.points
            for a, b in SKELETON_CONNECTIONS:
                if a in pts and b in pts:
                    pa = (int(pts[a][0]), int(pts[a][1]))
                    pb = (int(pts[b][0]), int(pts[b][1]))
                    cv2.line(frame, pa, pb, (0, 200, 140), 2)
            for name, (x, y, vis) in pts.items():
                if vis > 0.3:
                    cv2.circle(frame, (int(x), int(y)), 3, (240, 240, 240), -1)

        if idx in labeled_frames:
            m = labeled_frames[idx]
            wrist_key = f"{m.event.hand}_WRIST"
            if fl.detected and wrist_key in fl.points:
                wx, wy, _ = fl.points[wrist_key]
                cv2.circle(frame, (int(wx), int(wy)), 14, (60, 220, 255), 3)
            cv2.putText(frame, f"SWING #{m.event.index} - {m.zone}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 220, 255), 2, cv2.LINE_AA)

        writer.write(frame)
        idx += 1

    cap.release()
    writer.release()
