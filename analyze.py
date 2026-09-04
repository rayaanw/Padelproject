#!/usr/bin/env python
"""Rebote CLI: analyze a padel/racket-sport video for swing technique.

    python analyze.py sample_videos/man-playing-tennis-877.mp4

MVP scope (see PRD §04 / this session's scoping decisions):
  - heuristic swing detection (wrist-speed peaks), not a trained classifier
  - biomechanical metrics from pose math, not calibrated against real coaches yet
  - coaching feedback via a mock template, upgrading automatically to the
    live Claude API if ANTHROPIC_API_KEY is set in the environment
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rebote_lib.coach import generate_feedback, summarize
from rebote_lib.metrics import compute_all
from rebote_lib.pose_extraction import extract_pose_sequence
from rebote_lib.report import annotate_video, export_json, render_console
from rebote_lib.swing_detection import find_swings


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", help="path to a video file (mp4, mov, etc.)")
    p.add_argument("--hand", choices=["auto", "left", "right"], default="auto",
                   help="which wrist to track as the hitting arm (default: auto-detect)")
    p.add_argument("--model-complexity", type=int, choices=[0, 1, 2], default=1,
                   help="MediaPipe Pose complexity: 0=fastest, 2=most accurate (default: 1)")
    p.add_argument("--std-threshold", type=float, default=1.2,
                   help="swing-detection sensitivity: lower = more swings detected, "
                        "noisier (default: 1.2 standard deviations above mean speed)")
    p.add_argument("--min-gap", type=float, default=0.5,
                   help="minimum seconds between two detected swings (default: 0.5)")
    p.add_argument("--json-out", default=None, help="path for the JSON report (default: output/<name>_report.json)")
    p.add_argument("--no-video", action="store_true", help="skip writing the annotated overlay video")
    p.add_argument("--video-out", default=None, help="path for the annotated video (default: output/<name>_annotated.mp4)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"error: video not found: {video_path}", file=sys.stderr)
        return 1

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    json_out = args.json_out or str(out_dir / f"{video_path.stem}_report.json")
    video_out = args.video_out or str(out_dir / f"{video_path.stem}_annotated.mp4")

    print(f"Extracting pose from {video_path.name} ...", file=sys.stderr)
    seq = extract_pose_sequence(str(video_path), model_complexity=args.model_complexity)

    hand = None if args.hand == "auto" else args.hand.upper()
    hand_used, _speeds, events = find_swings(
        seq, hand=hand, std_threshold=args.std_threshold, min_gap_s=args.min_gap
    )

    metrics = compute_all(seq, events)
    summary = summarize(hand_used, seq.detection_rate, metrics)
    feedback, source = generate_feedback(summary)

    render_console(str(video_path), seq, metrics, summary, feedback, source)
    export_json(json_out, str(video_path), seq, metrics, summary, feedback, source)
    print(f"\nJSON report -> {json_out}", file=sys.stderr)

    if not args.no_video:
        print(f"Rendering annotated video -> {video_out} ...", file=sys.stderr)
        annotate_video(str(video_path), seq, metrics, video_out)
        print(f"Annotated video -> {video_out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
