from __future__ import annotations

from collections import Counter
from typing import Any

from ..contracts import ReviewBundle
from ..utils import utc_now_iso


def _primary_bpm(song_event_map: dict[str, Any]) -> float:
    hypotheses = list(song_event_map.get("tempo_hypotheses", []))
    if not hypotheses:
        return 0.0
    return float(hypotheses[0].get("bpm", 0.0) or 0.0)


def _section_counts(steps: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(step.get("section_label", "unknown") for step in steps)
    return dict(counts)


def _sequence_counts(steps: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(step.get("source_sequence", "unknown") for step in steps)
    return dict(counts)


def _sample_steps(steps: list[dict[str, Any]], count: int = 8) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for step in steps[:count]:
        samples.append(
            {
                "index": step.get("index"),
                "section_label": step.get("section_label"),
                "start_beat": step.get("start_beat"),
                "target_beats": step.get("target_beats"),
                "selected_unit_id": step.get("selected_unit_id"),
                "source_sequence": step.get("source_sequence"),
                "switch_reason": step.get("switch_reason", {}),
            }
        )
    return samples


def _retarget_hotspots(retarget_report: dict[str, Any], count: int = 3) -> list[dict[str, Any]]:
    reports = list(retarget_report.get("sequence_reports", []))
    ranked = sorted(
        reports,
        key=lambda item: float(item.get("joint_position_error", {}).get("mean") or 0.0),
        reverse=True,
    )
    hotspots: list[dict[str, Any]] = []
    for item in ranked[:count]:
        hotspots.append(
            {
                "dataset": item.get("dataset"),
                "sequence_id": item.get("sequence_id"),
                "status": item.get("status"),
                "joint_error_mean": item.get("joint_position_error", {}).get("mean"),
                "bone_angle_max": item.get("bone_angle_deg", {}).get("max"),
                "issues": item.get("issues", []),
            }
        )
    return hotspots


def _ordered_unique(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        ordered.append(value)
        seen.add(value)
    return ordered


def _format_counts(counts: dict[str, Any]) -> str:
    if not counts:
        return "n/a"
    return ", ".join(f"{key}={value}" for key, value in counts.items())


def _action_items(bundle: dict[str, Any]) -> list[str]:
    items: list[str] = []
    song = bundle.get("song_summary", {})
    retarget = bundle.get("retarget_summary", {})
    summary = retarget.get("summary", {})
    hotspots = retarget.get("hotspots", [])

    if song.get("manual_review_required"):
        items.append("Apply manual beat, phrase, and section overrides before trusting this song for automatic choreography.")

    if not retarget.get("backend", {}).get("available", False):
        items.append("Install or configure Blender locally so preview jobs can execute instead of remaining dry-run handoff artifacts.")

    worst_joint_error = float(summary.get("worst_joint_error_mean") or 0.0)
    if hotspots and worst_joint_error >= 2.0:
        worst = hotspots[0]
        items.append(
            f"Inspect retarget hotspot `{worst.get('dataset')}:{worst.get('sequence_id')}` first because its joint error mean is still {worst_joint_error:.3f}."
        )

    return items


def build_review_bundle(
    song_event_map: dict[str, Any],
    motion_library_showcase: dict[str, Any],
    choreography_plan: dict[str, Any],
    retarget_report: dict[str, Any],
) -> ReviewBundle:
    steps = list(choreography_plan.get("steps", []))
    bundle_id = f"{song_event_map['song_id']}_review_bundle"

    song_summary = {
        "analysis_mode": song_event_map.get("analysis_mode"),
        "manual_review_required": song_event_map.get("manual_review_required"),
        "override_source": song_event_map.get("override_source"),
        "applied_override_keys": song_event_map.get("applied_override_keys", []),
        "duration_sec": song_event_map.get("duration_sec"),
        "beats_per_bar": song_event_map.get("beats_per_bar"),
        "beat_count": len(song_event_map.get("beats", [])),
        "phrase_count": len(song_event_map.get("phrases", [])),
        "section_labels": [section.get("label") for section in song_event_map.get("sections", [])],
        "primary_bpm": _primary_bpm(song_event_map),
        "confidence": song_event_map.get("confidence", {}),
    }

    library_summary = {
        "library_id": motion_library_showcase.get("library_id"),
        "counts": motion_library_showcase.get("counts", {}),
        "sampled_units": motion_library_showcase.get("sampled_units", [])[:6],
    }

    choreography_summary = {
        "step_count": len(steps),
        "section_counts": _section_counts(steps),
        "source_sequence_counts": _sequence_counts(steps),
        "sample_steps": _sample_steps(steps),
    }

    retarget_summary = {
        "status": retarget_report.get("status"),
        "backend": retarget_report.get("backend", {}),
        "summary": retarget_report.get("summary", {}),
        "hotspots": _retarget_hotspots(retarget_report),
    }

    highlights: list[str] = []
    if song_summary["manual_review_required"]:
        highlights.append("Song analysis still requests manual review before trusting automatic choreography.")
    else:
        highlights.append(f"Song timing is currently in `{song_summary['analysis_mode']}` mode and does not force manual review.")

    counts = library_summary.get("counts", {})
    travel_counts = counts.get("travel", {})
    if travel_counts:
        highlights.append(
            f"Motion library now includes {counts.get('unit_count', 0)} units with travel mix: "
            f"{travel_counts.get('traveling', 0)} traveling, {travel_counts.get('localized', 0)} localized, {travel_counts.get('stationary', 0)} stationary."
        )

    highlights.append(
        f"Current choreography plan spans {choreography_summary['step_count']} steps across sections: "
        f"{', '.join(f'{label}={count}' for label, count in choreography_summary['section_counts'].items())}."
    )

    hotspots = retarget_summary["hotspots"]
    if hotspots:
        worst = hotspots[0]
        highlights.append(
            f"Current worst retarget hotspot is {worst['dataset']}:{worst['sequence_id']} "
            f"(joint_error_mean={worst['joint_error_mean']}, bone_angle_max={worst['bone_angle_max']})."
        )
    if not retarget_summary["backend"].get("available", False):
        highlights.append("Blender backend is still unavailable locally, so preview execution remains a dry handoff artifact.")

    return ReviewBundle(
        schema_version=1,
        bundle_id=bundle_id,
        song_id=str(song_event_map["song_id"]),
        plan_id=str(choreography_plan["plan_id"]),
        song_summary=song_summary,
        library_summary=library_summary,
        choreography_summary=choreography_summary,
        retarget_summary=retarget_summary,
        highlights=highlights,
        generated_at_utc=utc_now_iso(),
    )


def build_review_bundle_markdown(bundle: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Review Bundle: {bundle['song_id']}")
    lines.append("")
    lines.append(f"- generated_at_utc: `{bundle.get('generated_at_utc')}`")
    lines.append(f"- plan_id: `{bundle.get('plan_id')}`")
    lines.append("")
    lines.append("## Highlights")
    for item in bundle.get("highlights", []):
        lines.append(f"- {item}")
    lines.append("")

    action_items = _action_items(bundle)
    if action_items:
        lines.append("## Action Items")
        for item in action_items:
            lines.append(f"- {item}")
        lines.append("")

    song = bundle.get("song_summary", {})
    ordered_sections = _ordered_unique(list(song.get("section_labels", [])))
    confidence = song.get("confidence", {})
    lines.append("## Song Summary")
    lines.append(f"- analysis_mode: `{song.get('analysis_mode')}`")
    lines.append(f"- manual_review_required: `{song.get('manual_review_required')}`")
    lines.append(f"- duration_sec: `{song.get('duration_sec')}`")
    lines.append(f"- primary_bpm: `{song.get('primary_bpm')}`")
    lines.append(f"- beats_per_bar: `{song.get('beats_per_bar')}`")
    lines.append(f"- beat_count: `{song.get('beat_count')}`")
    lines.append(f"- phrase_count: `{song.get('phrase_count')}`")
    lines.append(f"- section_flow: `{' -> '.join(ordered_sections) if ordered_sections else 'n/a'}`")
    lines.append(
        "- confidence: "
        f"overall=`{confidence.get('overall')}` "
        f"beat_stability=`{confidence.get('beat_stability')}` "
        f"accent_clarity=`{confidence.get('accent_clarity')}` "
        f"phrase_confidence=`{confidence.get('phrase_confidence')}` "
        f"section_confidence=`{confidence.get('section_confidence')}`"
    )
    lines.append("")

    library = bundle.get("library_summary", {})
    counts = library.get("counts", {})
    lines.append("## Library Summary")
    lines.append(f"- library_id: `{library.get('library_id')}`")
    lines.append(f"- unit_count: `{counts.get('unit_count')}`")
    lines.append(f"- energy_mix: `{_format_counts(counts.get('energy', {}))}`")
    lines.append(f"- travel_mix: `{_format_counts(counts.get('travel', {}))}`")
    lines.append(f"- facing_mix: `{_format_counts(counts.get('facing_change', {}))}`")
    lines.append(f"- support_mix: `{_format_counts(counts.get('support_state', {}))}`")
    lines.append(f"- body_focus_mix: `{_format_counts(counts.get('body_focus', {}))}`")
    lines.append("- sampled_units:")
    for unit in library.get("sampled_units", [])[:4]:
        lines.append(
            f"  - `{unit.get('unit_id')}` "
            f"energy=`{unit.get('energy')}` "
            f"travel=`{unit.get('travel')}` "
            f"facing=`{unit.get('facing_change')}` "
            f"support=`{', '.join(unit.get('foot_contact', []))}` "
            f"body_focus=`{', '.join(unit.get('body_focus', []))}`"
        )
    lines.append("")

    choreography = bundle.get("choreography_summary", {})
    section_counts = choreography.get("section_counts", {})
    sequence_counts = choreography.get("source_sequence_counts", {})
    lines.append("## Choreography Summary")
    lines.append(f"- step_count: `{choreography.get('step_count')}`")
    lines.append(f"- section_counts: `{_format_counts(section_counts)}`")
    lines.append(f"- source_sequence_counts: `{_format_counts(sequence_counts)}`")
    lines.append("- sample_steps:")
    for step in choreography.get("sample_steps", [])[:6]:
        switch_reason = step.get("switch_reason", {})
        lines.append(
            f"  - step `{step.get('index')}` "
            f"section=`{step.get('section_label')}` "
            f"unit=`{step.get('selected_unit_id')}` "
            f"sequence=`{step.get('source_sequence')}` "
            f"energy=`{switch_reason.get('target_energy')}` "
            f"travel=`{switch_reason.get('target_travel')}` "
            f"compatible=`{switch_reason.get('compatible_from_previous')}`"
        )
    lines.append("")

    retarget = bundle.get("retarget_summary", {})
    summary = retarget.get("summary", {})
    lines.append("## Retarget Summary")
    lines.append(f"- status: `{retarget.get('status')}`")
    lines.append(f"- backend: `{retarget.get('backend')}`")
    lines.append(
        f"- worst_metrics: `joint_error_mean={summary.get('worst_joint_error_mean')}, "
        f"bone_angle_max={summary.get('worst_bone_angle_max')}`"
    )
    lines.append(f"- status_counts: `{_format_counts(summary.get('status_counts', {}))}`")
    lines.append("- hotspots:")
    for hotspot in retarget.get("hotspots", []):
        lines.append(
            f"  - `{hotspot.get('dataset')}:{hotspot.get('sequence_id')}` "
            f"status=`{hotspot.get('status')}` "
            f"joint_error_mean=`{hotspot.get('joint_error_mean')}` "
            f"bone_angle_max=`{hotspot.get('bone_angle_max')}` "
            f"issues=`{', '.join(hotspot.get('issues', [])) or 'n/a'}`"
        )
    lines.append("")
    return "\n".join(lines) + "\n"
