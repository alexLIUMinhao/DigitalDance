#!/usr/bin/env python3
"""Shared helpers for the isolated motion-base toolchain."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
MOTION_BASE_ROOT = REPO_ROOT / "motion_base"
CONFIG_ROOT = MOTION_BASE_ROOT / "config"
INTAKE_ROOT = MOTION_BASE_ROOT / "intake"
LIBRARY_ROOT = MOTION_BASE_ROOT / "library"
PREVIEW_ROOT = MOTION_BASE_ROOT / "preview" / "runtime_preview"
REVIEW_ROOT = MOTION_BASE_ROOT / "review"
VALIDATION_ROOT = MOTION_BASE_ROOT / "validation"
TAXONOMY_PATH = MOTION_BASE_ROOT / "taxonomy" / "controlled_vocab.json"
LOCAL_ASSET_CONFIG_PATH = CONFIG_ROOT / "asset_roots.local.json"
EXAMPLE_ASSET_CONFIG_PATH = CONFIG_ROOT / "asset_roots.example.json"
MOTION_INDEX_PATH = LIBRARY_ROOT / "motion_index.json"
REVIEWS_PATH = LIBRARY_ROOT / "reviews.json"
INTAKE_QUEUE_PATH = INTAKE_ROOT / "intake_queue.json"
SOURCE_CATALOG_PATH = INTAKE_ROOT / "source_catalog.json"
DATASET_CATALOG_PATH = INTAKE_ROOT / "dataset_catalog.json"
CANDIDATE_REVIEW_PATH = REVIEW_ROOT / "candidate_review.json"
REVIEW_FEED_PATH = REVIEW_ROOT / "review_feed.json"
CANDIDATE_METRICS_PATH = VALIDATION_ROOT / "candidate_metrics.json"

UNITY_REVIEW_CACHE_RELATIVE = Path("Assets") / "MotionBaseReviewCache"

MOTION_REQUIRED_FIELDS = (
    "motionId",
    "displayName",
    "approvedFbxRelPath",
    "sourceKind",
    "sourceRef",
    "licenseTier",
    "qualityTier",
    "styleFamily",
    "styleSubstyle",
    "metaAction",
    "energyBand",
    "preferredSegments",
    "nativeBpm",
    "phraseBeats",
    "entryOffsetsBeats",
    "sliceBeatsOptions",
    "transitionProfile",
    "varietyGroup",
    "role",
    "notes",
)

REVIEW_REQUIRED_FIELDS = (
    "motionId",
    "reviewStatus",
    "loopSeam",
    "footStability",
    "styleClarity",
    "tempoTolerance",
    "reviewer",
    "issues",
)

ENERGY_BANDS = {"low_energy", "mid_energy", "high_energy"}
ROLES = {"loop", "accent"}
ALLOWED_PHRASE_BEATS = {4, 8, 16}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".webm", ".mkv"}
MOTION_EXTENSIONS = {".fbx", ".bvh"}


@dataclass
class ValidationReport:
    errors: List[str]
    warnings: List[str]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def load_taxonomy() -> Dict[str, Any]:
    return load_json(TAXONOMY_PATH)


def load_library() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return load_json(MOTION_INDEX_PATH), load_json(REVIEWS_PATH)


def default_intake_queue() -> Dict[str, Any]:
    return {"schemaVersion": 1, "jobs": []}


def default_source_catalog() -> Dict[str, Any]:
    return {"schemaVersion": 1, "generatedAtUtc": utc_now_iso(), "sources": []}


def default_dataset_catalog() -> Dict[str, Any]:
    return {"schemaVersion": 1, "generatedAtUtc": utc_now_iso(), "entries": []}


def default_candidate_review() -> Dict[str, Any]:
    return {"schemaVersion": 1, "entries": []}


def default_review_feed() -> Dict[str, Any]:
    return {"schemaVersion": 1, "generatedAtUtc": utc_now_iso(), "entries": []}


def default_candidate_metrics() -> Dict[str, Any]:
    return {"schemaVersion": 1, "generatedAtUtc": utc_now_iso(), "metrics": []}


def load_or_default(path: Path, factory: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    if path.exists():
        return load_json(path)
    return factory()


def load_intake_queue() -> Dict[str, Any]:
    return load_or_default(INTAKE_QUEUE_PATH, default_intake_queue)


def load_source_catalog() -> Dict[str, Any]:
    return load_or_default(SOURCE_CATALOG_PATH, default_source_catalog)


def load_dataset_catalog() -> Dict[str, Any]:
    return load_or_default(DATASET_CATALOG_PATH, default_dataset_catalog)


def load_candidate_review() -> Dict[str, Any]:
    return load_or_default(CANDIDATE_REVIEW_PATH, default_candidate_review)


def load_review_feed() -> Dict[str, Any]:
    return load_or_default(REVIEW_FEED_PATH, default_review_feed)


def load_candidate_metrics() -> Dict[str, Any]:
    return load_or_default(CANDIDATE_METRICS_PATH, default_candidate_metrics)


def load_asset_config() -> Dict[str, Any] | None:
    if LOCAL_ASSET_CONFIG_PATH.exists():
        return load_json(LOCAL_ASSET_CONFIG_PATH)
    return None


def resolve_asset_root(root_name: str) -> Path | None:
    config = load_asset_config()
    if not config:
        return None

    roots = config.get("roots", {}) or {}
    root_value = roots.get(root_name)
    if not root_value:
        return None

    return Path(root_value).expanduser()


def resolve_provider_path(provider: str, field: str) -> Path | None:
    config = load_asset_config()
    if not config:
        return None

    providers = config.get("providers", {}) or {}
    provider_data = providers.get(provider, {}) or {}
    value = provider_data.get(field)
    if not value:
        return None

    return Path(value).expanduser()


def resolve_template_path(field: str) -> Path | None:
    config = load_asset_config()
    if not config:
        return None

    templates = config.get("templates", {}) or {}
    value = templates.get(field)
    if not value:
        return None

    return Path(value).expanduser()


def resolve_external_assets_root() -> Path | None:
    for root_name in ("rawFbx", "sourceVideo", "extractedMotion", "approvedFbx", "previewCache"):
        root = resolve_asset_root(root_name)
        if root is not None:
            return root.parent
    return None


def resolve_dataset_storage_root(dataset_name: str = "", bucket: str = "") -> Path | None:
    assets_root = resolve_external_assets_root()
    if assets_root is None:
        return None

    root = assets_root / "datasets"
    if dataset_name:
        root = root / slugify(dataset_name)
    if bucket:
        root = root / bucket
    return root


def resolve_model_storage_root(model_family: str = "") -> Path | None:
    assets_root = resolve_external_assets_root()
    if assets_root is None:
        return None

    root = assets_root / "models"
    if model_family:
        root = root / slugify(model_family)
    return root


def asset_root_configured(root_name: str = "approvedFbx") -> bool:
    return resolve_asset_root(root_name) is not None


def resolve_asset_path(root_name: str, relative_path: str) -> Path | None:
    root = resolve_asset_root(root_name)
    if root is None or not relative_path:
        return None
    return root / relative_path


def resolve_external_fbx_path(approved_fbx_rel_path: str) -> Path | None:
    return resolve_asset_path("approvedFbx", approved_fbx_rel_path)


def is_safe_relative_path(value: str) -> bool:
    if not value or Path(value).is_absolute():
        return False

    path = Path(value)
    return all(part not in {"..", ""} for part in path.parts)


def slugify(value: str) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_") or "item"


def titleize_slug(value: str) -> str:
    return " ".join(part.capitalize() for part in slugify(value).split("_"))


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def motion_sort_key(item: Dict[str, Any]) -> str:
    return str(item.get("motionId", ""))


def job_sort_key(item: Dict[str, Any]) -> str:
    return str(item.get("jobId", ""))


def candidate_sort_key(item: Dict[str, Any]) -> str:
    return str(item.get("candidateId", ""))


def source_sort_key(item: Dict[str, Any]) -> str:
    return str(item.get("sourceId", ""))


def dataset_sort_key(item: Dict[str, Any]) -> str:
    return f"{item.get('datasetName', '')}::{item.get('sequenceId', '')}"


def motion_map(motion_index: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {record["motionId"]: record for record in motion_index.get("motions", [])}


def review_map(review_index: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {record["motionId"]: record for record in review_index.get("reviews", [])}


def intake_job_map(intake_index: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {record["jobId"]: record for record in intake_index.get("jobs", [])}


def candidate_review_map(review_index: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {record["candidateId"]: record for record in review_index.get("entries", [])}


def candidate_metrics_map(metrics_index: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {record["candidateId"]: record for record in metrics_index.get("metrics", [])}


def source_catalog_map(source_catalog: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {record["sourceId"]: record for record in source_catalog.get("sources", [])}


def dataset_catalog_map(dataset_catalog: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        f"{record.get('datasetName', '')}::{record.get('sequenceId', '')}": record
        for record in dataset_catalog.get("entries", [])
    }


def upsert_records(
    existing: Iterable[Dict[str, Any]],
    incoming: Iterable[Dict[str, Any]],
    key_field: str,
    sort_key: Callable[[Dict[str, Any]], str],
) -> List[Dict[str, Any]]:
    merged = {record[key_field]: dict(record) for record in existing}
    for record in incoming:
        merged[record[key_field]] = dict(record)
    return sorted(merged.values(), key=sort_key)


def discover_style_family(path_like: str, taxonomy: Dict[str, Any]) -> str:
    normalized = slugify(path_like)
    for family in taxonomy.get("styleFamilies", []):
        if family in normalized:
            return family
    return ""


def discover_style_substyle(path_like: str, style_family: str) -> str:
    if not style_family:
        return ""
    parts = [slugify(part) for part in Path(path_like).parts]
    for index, part in enumerate(parts):
        if part == style_family and index + 1 < len(parts):
            return parts[index + 1]
    return ""


def discover_meta_action(path_like: str, taxonomy: Dict[str, Any]) -> str:
    normalized = slugify(path_like)
    aliases = {
        "basic_step": ("basic", "step", "walk", "groove"),
        "travel_step": ("travel", "move", "run", "cross"),
        "turn_phrase": ("turn", "spin", "twirl"),
        "arm_expression": ("arm", "hand", "sleeve", "fan"),
        "accent_hit": ("accent", "hit", "stab", "punch"),
        "pose_hold": ("pose", "hold", "freeze"),
    }
    for meta_action in taxonomy.get("metaActions", []):
        for alias in aliases.get(meta_action, (meta_action,)):
            if alias in normalized:
                return meta_action
    return ""


def infer_transition_profile(meta_action: str) -> str:
    if meta_action == "accent_hit":
        return "accent"
    if meta_action in {"turn_phrase", "pose_hold"}:
        return "bridge"
    return "style_locked"


def infer_energy_band(meta_action: str) -> str:
    if meta_action in {"pose_hold", "arm_expression"}:
        return "low_energy"
    if meta_action == "accent_hit":
        return "high_energy"
    return "mid_energy"


def infer_preferred_segments(meta_action: str, energy_band: str) -> List[str]:
    if meta_action == "pose_hold":
        return ["intro", "outro"]
    if meta_action == "arm_expression":
        return ["verse", "instrumental", "outro"]
    if energy_band == "high_energy":
        return ["chorus", "instrumental"]
    return ["verse", "chorus"]


def infer_role(meta_action: str, transition_profile: str) -> str:
    if meta_action == "accent_hit" or transition_profile == "accent":
        return "accent"
    return "loop"


def infer_source_kind(source_lane: str) -> str:
    if source_lane == "video_rokoko":
        return "generated_from_video"
    if source_lane in {"dataset_smpl", "dataset_json"}:
        return "dataset_motion"
    return "curated_fbx"


def make_source_id(provider: str, remote_asset_id: str, fallback_hint: str = "") -> str:
    if remote_asset_id:
        return f"{slugify(provider)}__{slugify(remote_asset_id)}"
    return f"{slugify(provider)}__{slugify(fallback_hint)}"


def make_dataset_entry_id(dataset_name: str, sequence_id: str) -> str:
    return f"{slugify(dataset_name)}__{slugify(sequence_id)}"


def make_job_id(source_lane: str, relative_path: str) -> str:
    return f"{source_lane}__{slugify(relative_path)}"


def make_candidate_id(base_motion_id: str, phrase_beats: int, entry_offset_beats: int, slice_index: int) -> str:
    return f"{base_motion_id}_p{phrase_beats:02d}_o{entry_offset_beats:02d}_{slice_index:02d}"


def build_motion_record_from_candidate(candidate_review: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "motionId": candidate_review["motionId"],
        "displayName": candidate_review["displayName"],
        "approvedFbxRelPath": candidate_review["candidateFbxRelPath"],
        "sourceKind": infer_source_kind(candidate_review["sourceLane"]),
        "sourceRef": f"candidate_review/{candidate_review['candidateId']}",
        "licenseTier": candidate_review["licenseTier"],
        "qualityTier": "production",
        "styleFamily": candidate_review["targetStyleFamily"],
        "styleSubstyle": candidate_review["targetStyleSubstyle"],
        "metaAction": candidate_review["expectedMetaAction"],
        "energyBand": candidate_review["energyBand"],
        "preferredSegments": list(candidate_review["preferredSegments"]),
        "nativeBpm": float(candidate_review["nativeBpm"]),
        "phraseBeats": int(candidate_review["phraseBeats"]),
        "entryOffsetsBeats": list(candidate_review["entryOffsetsBeats"]),
        "sliceBeatsOptions": list(candidate_review["sliceBeatsOptions"]),
        "transitionProfile": candidate_review["transitionProfile"],
        "varietyGroup": candidate_review["varietyGroup"],
        "role": candidate_review["role"],
        "notes": candidate_review.get("notes", ""),
    }


def build_review_record_from_candidate(candidate_review: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "motionId": candidate_review["motionId"],
        "reviewStatus": candidate_review["reviewStatus"],
        "loopSeam": candidate_review["loopSeam"],
        "footStability": candidate_review["footStability"],
        "styleClarity": candidate_review["styleClarity"],
        "tempoTolerance": candidate_review["tempoTolerance"],
        "reviewer": candidate_review["reviewer"],
        "issues": list(candidate_review["issues"]),
        "reviewedAtUtc": candidate_review.get("savedAtUtc", utc_now_iso()),
    }


def _missing_fields(record: Dict[str, Any], required_fields: Iterable[str]) -> List[str]:
    return [field for field in required_fields if field not in record]


def validate_motion_record(record: Dict[str, Any], taxonomy: Dict[str, Any]) -> ValidationReport:
    errors: List[str] = []
    warnings: List[str] = []
    missing = _missing_fields(record, MOTION_REQUIRED_FIELDS)
    if missing:
        errors.append(f"{record.get('motionId', '<missing>')}: missing fields: {', '.join(missing)}")
        return ValidationReport(errors, warnings)

    motion_id = str(record["motionId"])
    if not motion_id:
        errors.append("motion record has empty motionId")

    if not is_safe_relative_path(str(record["approvedFbxRelPath"])):
        errors.append(f"{motion_id}: approvedFbxRelPath must be a safe repo-independent relative path")

    if str(record["styleFamily"]) not in set(taxonomy.get("styleFamilies", [])):
        errors.append(f"{motion_id}: invalid styleFamily '{record['styleFamily']}'")

    if str(record["metaAction"]) not in set(taxonomy.get("metaActions", [])):
        errors.append(f"{motion_id}: invalid metaAction '{record['metaAction']}'")

    if str(record["transitionProfile"]) not in set(taxonomy.get("transitionProfiles", [])):
        errors.append(f"{motion_id}: invalid transitionProfile '{record['transitionProfile']}'")

    if str(record["sourceKind"]) not in set(taxonomy.get("sourceKinds", [])):
        errors.append(f"{motion_id}: invalid sourceKind '{record['sourceKind']}'")

    if str(record["qualityTier"]) not in set(taxonomy.get("qualityTiers", [])):
        errors.append(f"{motion_id}: invalid qualityTier '{record['qualityTier']}'")

    if str(record["licenseTier"]) not in set(taxonomy.get("licenseTiers", [])):
        errors.append(f"{motion_id}: invalid licenseTier '{record['licenseTier']}'")

    if str(record["energyBand"]) not in ENERGY_BANDS:
        errors.append(f"{motion_id}: invalid energyBand '{record['energyBand']}'")

    if str(record["role"]) not in ROLES:
        errors.append(f"{motion_id}: invalid role '{record['role']}'")

    phrase_beats = int(record["phraseBeats"])
    if phrase_beats not in ALLOWED_PHRASE_BEATS:
        errors.append(f"{motion_id}: phraseBeats must be one of {sorted(ALLOWED_PHRASE_BEATS)}")

    preferred_segments = record["preferredSegments"]
    if not isinstance(preferred_segments, list) or not preferred_segments:
        errors.append(f"{motion_id}: preferredSegments must be a non-empty list")

    for field_name in ("entryOffsetsBeats", "sliceBeatsOptions"):
        values = record[field_name]
        if not isinstance(values, list) or not values:
            errors.append(f"{motion_id}: {field_name} must be a non-empty list")
            continue
        for value in values:
            if int(value) < 0:
                errors.append(f"{motion_id}: {field_name} values must be non-negative")

    native_bpm = float(record["nativeBpm"])
    if native_bpm <= 0:
        errors.append(f"{motion_id}: nativeBpm must be positive")

    resolved_path = resolve_external_fbx_path(str(record["approvedFbxRelPath"]))
    if resolved_path is not None and not resolved_path.exists():
        warnings.append(f"{motion_id}: external FBX not found at {resolved_path}")

    return ValidationReport(errors, warnings)


def validate_review_record(record: Dict[str, Any], taxonomy: Dict[str, Any]) -> ValidationReport:
    errors: List[str] = []
    warnings: List[str] = []
    missing = _missing_fields(record, REVIEW_REQUIRED_FIELDS)
    if missing:
        errors.append(f"{record.get('motionId', '<missing>')}: missing review fields: {', '.join(missing)}")
        return ValidationReport(errors, warnings)

    motion_id = str(record["motionId"])
    if str(record["reviewStatus"]) not in set(taxonomy.get("reviewStatuses", [])):
        errors.append(f"{motion_id}: invalid reviewStatus '{record['reviewStatus']}'")

    if str(record["loopSeam"]) not in set(taxonomy.get("loopSeamRatings", [])):
        errors.append(f"{motion_id}: invalid loopSeam '{record['loopSeam']}'")

    if str(record["footStability"]) not in set(taxonomy.get("footStabilityRatings", [])):
        errors.append(f"{motion_id}: invalid footStability '{record['footStability']}'")

    if str(record["styleClarity"]) not in set(taxonomy.get("styleClarityRatings", [])):
        errors.append(f"{motion_id}: invalid styleClarity '{record['styleClarity']}'")

    if str(record["tempoTolerance"]) not in set(taxonomy.get("tempoToleranceRatings", [])):
        errors.append(f"{motion_id}: invalid tempoTolerance '{record['tempoTolerance']}'")

    if not isinstance(record["issues"], list):
        errors.append(f"{motion_id}: issues must be a list")

    reviewer = str(record["reviewer"]).strip()
    if not reviewer:
        errors.append(f"{motion_id}: reviewer must be non-empty")

    if not str(record["reviewStatus"]).strip():
        errors.append(f"{motion_id}: reviewStatus must be non-empty")

    return ValidationReport(errors, warnings)
