"""Stage 4: turn structured swing metrics into coaching feedback.

Per the "LLM setup" decision, this defaults to a mock: it renders a
templated but data-grounded report so the pipeline is fully testable
without an API key. Set ANTHROPIC_API_KEY and it upgrades automatically
to a real Claude API call — same summary payload either way, so nothing
else in the pipeline needs to change when you add the key.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass

from .metrics import SwingMetrics

DEFAULT_MODEL = os.environ.get("REBOTE_MODEL", "claude-sonnet-5")


@dataclass
class SessionSummary:
    hand: str
    pose_detection_rate: float
    swing_count: int
    zone_counts: Counter
    avg_peak_speed: float
    split_step_rate: float
    fastest: SwingMetrics | None
    slowest: SwingMetrics | None
    lowest_contact: SwingMetrics | None


def summarize(hand: str, pose_detection_rate: float, metrics: list[SwingMetrics]) -> SessionSummary:
    if not metrics:
        return SessionSummary(hand, pose_detection_rate, 0, Counter(), 0.0, 0.0, None, None, None)

    zone_counts = Counter(m.zone for m in metrics)
    speeds = [m.event.peak_speed for m in metrics]
    with_contact = [m for m in metrics if m.contact_height_ratio is not None]

    return SessionSummary(
        hand=hand,
        pose_detection_rate=pose_detection_rate,
        swing_count=len(metrics),
        zone_counts=zone_counts,
        avg_peak_speed=sum(speeds) / len(speeds),
        split_step_rate=sum(m.split_step_detected for m in metrics) / len(metrics),
        fastest=max(metrics, key=lambda m: m.event.peak_speed),
        slowest=min(metrics, key=lambda m: m.event.peak_speed),
        lowest_contact=min(with_contact, key=lambda m: m.contact_height_ratio) if with_contact else None,
    )


def _fmt_t(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:04.1f}"


def _mock_feedback(summary: SessionSummary) -> str:
    """Deterministic, numbers-grounded stand-in for the real coaching call.
    Every figure quoted here comes from the actual computed metrics -
    it's a template, not a fabrication."""
    if summary.swing_count == 0:
        return ("No swings detected above the speed threshold. Either the clip has no "
                "hitting action, the camera was too far away for reliable wrist tracking, "
                "or the dominant-hand guess was wrong - try `--hand left` or `--hand right`.")

    lines = []
    top_zone, top_count = summary.zone_counts.most_common(1)[0]
    lines.append(
        f"Detected {summary.swing_count} swings on the {summary.hand.lower()} hand "
        f"(pose tracked cleanly on {summary.pose_detection_rate:.0%} of frames). "
        f"Most of your contact points ({top_count}/{summary.swing_count}) fell in the "
        f"{top_zone.lower()}."
    )

    if summary.split_step_rate < 0.4:
        lines.append(
            f"Split-step was only picked up before {summary.split_step_rate:.0%} of swings. "
            f"That's the single fix worth drilling first: a hop timed to your opponent's "
            f"contact - not your own recovery - gets your weight loaded before the ball arrives "
            f"instead of after."
        )
    else:
        lines.append(
            f"Split-step showed up before {summary.split_step_rate:.0%} of swings - good base "
            f"habit, keep it."
        )

    if summary.slowest and summary.fastest and summary.fastest.event.peak_speed > 0:
        ratio = summary.slowest.event.peak_speed / summary.fastest.event.peak_speed
        if ratio < 0.6:
            lines.append(
                f"Swing #{summary.slowest.event.index} at {_fmt_t(summary.slowest.event.t)} was "
                f"noticeably slower than your fastest (swing #{summary.fastest.event.index}, "
                f"{_fmt_t(summary.fastest.event.t)}) - {summary.slowest.event.peak_speed:.1f} vs "
                f"{summary.fastest.event.peak_speed:.1f} torso-lengths/sec. Worth reviewing that "
                f"rep on tape to see if it was a defensive shot or a mishit."
            )

    if summary.lowest_contact and summary.lowest_contact.contact_height_ratio < -0.45:
        lines.append(
            f"Your lowest contact point (swing #{summary.lowest_contact.event.index}, "
            f"{_fmt_t(summary.lowest_contact.event.t)}) was well below the belt - fine if that's "
            f"a defensive dig, worth flagging if it was meant to be an offensive shot."
        )

    return " ".join(lines)


def _live_feedback(summary: SessionSummary, model: str = DEFAULT_MODEL) -> str:
    import anthropic  # imported lazily so the mock path never needs the package configured

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    payload = {
        "hand": summary.hand,
        "pose_detection_rate": round(summary.pose_detection_rate, 3),
        "swing_count": summary.swing_count,
        "zone_counts": dict(summary.zone_counts),
        "avg_peak_speed_torso_lengths_per_s": round(summary.avg_peak_speed, 2),
        "split_step_rate": round(summary.split_step_rate, 3),
        "fastest_swing": summary.fastest and {
            "index": summary.fastest.event.index, "t": round(summary.fastest.event.t, 2),
            "peak_speed": round(summary.fastest.event.peak_speed, 2),
        },
        "slowest_swing": summary.slowest and {
            "index": summary.slowest.event.index, "t": round(summary.slowest.event.t, 2),
            "peak_speed": round(summary.slowest.event.peak_speed, 2),
        },
    }
    response = client.messages.create(
        model=model,
        max_tokens=400,
        system=(
            "You are a padel coach reviewing a player's session, reconstructed from pose-"
            "tracking data on a phone-recorded clip. You get kinematic metrics, not shot names "
            "a trained classifier would give you - speak about contact height, timing, and "
            "split-step, not paddle names you can't actually see. Give 2-3 short, specific, "
            "encouraging-but-honest notes a player could act on next session. No headers, no "
            "bullet points - plain prose."
        ),
        messages=[{"role": "user", "content": str(payload)}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def generate_feedback(summary: SessionSummary) -> tuple[str, str]:
    """Returns (feedback_text, source) where source is 'claude-api' or 'mock'."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return _live_feedback(summary), "claude-api"
        except Exception as exc:  # noqa: BLE001 - surface as a downgrade, not a crash
            return (
                f"[Claude API call failed ({exc}); showing mock feedback instead]\n"
                + _mock_feedback(summary)
            ), "mock (api error)"
    return _mock_feedback(summary), "mock"
