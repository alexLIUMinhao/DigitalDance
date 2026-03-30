#!/usr/bin/env python3
"""Scan downloaded dance datasets and build a normalized dataset catalog."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from common import (
    DATASET_CATALOG_PATH,
    default_dataset_catalog,
    load_dataset_catalog,
    make_dataset_entry_id,
    resolve_dataset_storage_root,
    slugify,
    titleize_slug,
    upsert_records,
    utc_now_iso,
    write_json,
)


AIST_STYLE_MAP = {
    "gBR": "break",
    "gPO": "popping",
    "gLO": "locking",
    "gMH": "middle_hip_hop",
    "gLH": "la_hip_hop",
    "gHO": "house",
    "gWA": "waack",
    "gKR": "krump",
    "gJS": "street_jazz",
    "gJB": "ballet_jazz",
}

DATASET_META = {
    "aistpp": {
        "sourcePageUrl": "https://google.github.io/aistplusplus_dataset/download.html",
        "license": "AIST++ Terms of Use",
        "fps": 60.0,
        "sourceFormat": "smpl_pkl",
    },
    "finedance": {
        "sourcePageUrl": "https://github.com/li-ronghui/FineDance",
        "license": "FineDance research-only dataset",
        "fps": 30.0,
        "sourceFormat": "smplh_npy",
    },
    "phantomdance": {
        "sourcePageUrl": "https://github.com/libuyu/PhantomDanceDataset",
        "license": "PhantomDance research dataset",
        "fps": 30.0,
        "sourceFormat": "json_quat_smpl24",
    },
    "aioz_gdance": {
        "sourcePageUrl": "https://huggingface.co/datasets/aiozai/AIOZ-GDANCE",
        "license": "AIOZ-GDANCE non-commercial scientific research license",
        "fps": 30.0,
        "sourceFormat": "smpl_pkl",
    },
    "souldance": {
        "sourcePageUrl": "https://github.com/xjli360/SoulDance-Official",
        "license": "SoulDance EULA required",
        "fps": 30.0,
        "sourceFormat": "feature_only",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DATASET_CATALOG_PATH, help="Output dataset catalog path.")
    parser.add_argument("--datasets-root", type=Path, help="Override datasets root. Defaults to sibling of rawFbx root.")
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASET_META.keys()),
        help="Optional dataset name filter. Can be passed multiple times.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print summary without writing the catalog.")
    return parser.parse_args()


def choose_datasets_root(args: argparse.Namespace) -> Path | None:
    if args.datasets_root:
        return args.datasets_root.expanduser()
    return resolve_dataset_storage_root()


def relative_or_empty(root: Path, path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return path.relative_to(root).as_posix()


def style_payload(
    style_family: str = "contemporary",
    style_substyle: str = "",
    expected_meta_action: str = "basic_step",
    tags: Iterable[str] = (),
) -> Dict[str, Any]:
    return {
        "styleFamily": style_family,
        "styleSubstyle": style_substyle,
        "expectedMetaAction": expected_meta_action,
        "tags": [str(tag) for tag in tags if str(tag).strip()],
    }


def base_entry(
    dataset_name: str,
    sequence_id: str,
    raw_root: Path,
    motion_path: Path,
    music_path: Path | None,
    label_path: Path | None,
    style_labels: Dict[str, Any],
) -> Dict[str, Any]:
    meta = DATASET_META[dataset_name]
    return {
        "entryId": make_dataset_entry_id(dataset_name, sequence_id),
        "datasetName": dataset_name,
        "sequenceId": sequence_id,
        "motionPath": relative_or_empty(raw_root, motion_path),
        "musicPath": relative_or_empty(raw_root, music_path),
        "labelPath": relative_or_empty(raw_root, label_path),
        "fps": float(meta["fps"]),
        "sourceFormat": meta["sourceFormat"],
        "license": meta["license"],
        "sourcePageUrl": meta["sourcePageUrl"],
        "styleLabels": style_labels,
        "targetStyleFamily": str(style_labels.get("styleFamily", "") or ""),
        "targetStyleSubstyle": str(style_labels.get("styleSubstyle", "") or ""),
        "expectedMetaAction": str(style_labels.get("expectedMetaAction", "") or ""),
        "convertedMotionRelPath": "",
        "conversionStatus": "pending",
        "convertedAtUtc": "",
        "notes": "",
    }


def merge_existing(existing: Dict[str, Any] | None, incoming: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(incoming)
    if not existing:
        return merged

    for key in ("convertedMotionRelPath", "conversionStatus", "convertedAtUtc", "notes"):
        if existing.get(key) not in (None, ""):
            merged[key] = existing[key]
    return merged


def first_existing(*candidates: Path) -> Path | None:
    return next((candidate for candidate in candidates if candidate.exists()), None)


def parse_finedance_label(path: Path | None) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(payload, dict):
        return {}

    coarse = (
        payload.get("coarse_style")
        or payload.get("coarseStyle")
        or payload.get("style")
        or payload.get("style1")
        or ""
    )
    fine = (
        payload.get("fine_style")
        or payload.get("fineStyle")
        or payload.get("genre")
        or payload.get("style2")
        or ""
    )
    song_name = payload.get("song_name") or payload.get("songName") or payload.get("music") or ""
    return {
        "coarseStyle": str(coarse or ""),
        "fineStyle": str(fine or ""),
        "songName": str(song_name or ""),
    }


def scan_aistpp(raw_root: Path, existing: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for motion_path in sorted(path for path in raw_root.rglob("*.pkl") if path.is_file()):
        sequence_id = motion_path.stem
        genre_code = sequence_id.split("_", 1)[0]
        style_substyle = AIST_STYLE_MAP.get(genre_code, slugify(genre_code or "aistpp"))
        style_labels = style_payload(style_substyle=style_substyle, tags=[genre_code, style_substyle])
        key = make_dataset_entry_id("aistpp", sequence_id)
        entries.append(
            merge_existing(
                existing.get(key),
                base_entry("aistpp", sequence_id, raw_root, motion_path, None, None, style_labels),
            )
        )
    return entries


def scan_finedance(raw_root: Path, existing: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for motion_path in sorted(path for path in raw_root.rglob("*.npy") if path.is_file() and "motion" in path.parts):
        sequence_id = motion_path.stem
        music_path = first_existing(
            raw_root / "music_wav" / f"{sequence_id}.wav",
            raw_root / "music" / f"{sequence_id}.wav",
        )
        label_path = first_existing(
            raw_root / "label_json" / f"{sequence_id}.json",
            raw_root / "labels" / f"{sequence_id}.json",
        )
        label_info = parse_finedance_label(label_path)
        fine_style = slugify(label_info.get("fineStyle", "") or label_info.get("coarseStyle", "") or "finedance")
        tags = [label_info.get("coarseStyle", ""), label_info.get("fineStyle", ""), label_info.get("songName", "")]
        style_labels = style_payload(style_substyle=fine_style, tags=tags)
        key = make_dataset_entry_id("finedance", sequence_id)
        entry = base_entry("finedance", sequence_id, raw_root, motion_path, music_path, label_path, style_labels)
        if label_info:
            entry["labelSummary"] = label_info
        entries.append(merge_existing(existing.get(key), entry))
    return entries


def phantomdance_style_hint(sequence_id: str, motion_path: Path) -> str:
    parent_hint = slugify(motion_path.parent.name)
    prefix_match = re.match(r"[A-Za-z]+", sequence_id)
    prefix_hint = slugify(prefix_match.group(0)) if prefix_match else ""
    for candidate in (parent_hint, prefix_hint, "phantomdance"):
        if candidate and candidate not in {"json", "motion", "motions", "rawdata", "raw_data"}:
            return candidate
    return "phantomdance"


def looks_like_phantomdance_motion(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    return all(key in payload for key in ("bone_name", "root_positions", "rotations"))


def scan_phantomdance(raw_root: Path, existing: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    wav_by_stem = {path.stem: path for path in raw_root.rglob("*.wav") if path.is_file()}
    for motion_path in sorted(path for path in raw_root.rglob("*.json") if path.is_file()):
        if not looks_like_phantomdance_motion(motion_path):
            continue
        sequence_id = motion_path.stem
        music_path = wav_by_stem.get(sequence_id)
        style_substyle = phantomdance_style_hint(sequence_id, motion_path)
        style_labels = style_payload(style_substyle=style_substyle, tags=[style_substyle])
        key = make_dataset_entry_id("phantomdance", sequence_id)
        entries.append(
            merge_existing(
                existing.get(key),
                base_entry("phantomdance", sequence_id, raw_root, motion_path, music_path, None, style_labels),
            )
        )
    return entries


def scan_aioz_gdance(raw_root: Path, existing: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for motion_path in sorted(path for path in raw_root.rglob("*.pkl") if path.is_file() and "motions_smpl" in path.parts):
        sequence_id = motion_path.stem
        music_path = first_existing(raw_root / "musics" / f"{sequence_id}.wav")
        style_labels = style_payload(style_substyle="group_dance", tags=["group_dance"])
        key = make_dataset_entry_id("aioz_gdance", sequence_id)
        entries.append(
            merge_existing(
                existing.get(key),
                base_entry("aioz_gdance", sequence_id, raw_root, motion_path, music_path, None, style_labels),
            )
        )
    return entries


def scan_souldance(raw_root: Path, existing: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for motion_path in sorted(path for path in raw_root.rglob("*.npy") if path.is_file()):
        sequence_id = motion_path.stem
        style_labels = style_payload(style_substyle="souldance", tags=["feature_only"])
        key = make_dataset_entry_id("souldance", sequence_id)
        entries.append(
            merge_existing(
                existing.get(key),
                base_entry("souldance", sequence_id, raw_root, motion_path, None, None, style_labels),
            )
        )
    return entries


SCAN_HANDLERS = {
    "aistpp": scan_aistpp,
    "finedance": scan_finedance,
    "phantomdance": scan_phantomdance,
    "aioz_gdance": scan_aioz_gdance,
    "souldance": scan_souldance,
}


def main() -> int:
    args = parse_args()
    datasets_root = choose_datasets_root(args)
    if datasets_root is None:
        print("ERROR: datasets root is not configured. Configure rawFbx or pass --datasets-root.")
        return 1

    existing_catalog = load_dataset_catalog() if args.output.exists() else default_dataset_catalog()
    existing_by_id = {record["entryId"]: record for record in existing_catalog.get("entries", [])}
    requested = args.dataset or sorted(SCAN_HANDLERS.keys())

    scanned_entries: List[Dict[str, Any]] = []
    counts: List[str] = []
    for dataset_name in requested:
        raw_root = datasets_root / dataset_name / "raw"
        if not raw_root.exists():
            continue
        entries = SCAN_HANDLERS[dataset_name](raw_root, existing_by_id)
        scanned_entries.extend(entries)
        counts.append(f"{dataset_name}={len(entries)}")

    merged_entries = upsert_records(
        existing_catalog.get("entries", []),
        scanned_entries,
        "entryId",
        lambda item: f"{item.get('datasetName', '')}::{item.get('sequenceId', '')}",
    )
    payload = {
        "schemaVersion": 1,
        "generatedAtUtc": utc_now_iso(),
        "entries": merged_entries,
    }

    if args.dry_run:
        print(f"dry-run dataset entries={len(merged_entries)} ({', '.join(counts)})")
        for entry in merged_entries[:20]:
            print(
                f"[{entry['datasetName']}] {titleize_slug(entry['sequenceId'])} "
                f"format={entry['sourceFormat']} converted={entry.get('conversionStatus', 'pending')}"
            )
        return 0

    write_json(args.output, payload)
    print(f"written dataset entries={len(merged_entries)} to {args.output} ({', '.join(counts)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
