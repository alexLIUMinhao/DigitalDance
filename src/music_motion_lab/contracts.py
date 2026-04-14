from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SongEventMap:
    schema_version: int
    song_id: str
    source_audio_path: str
    duration_sec: float
    beats_per_bar: int
    tempo_hypotheses: list[dict[str, Any]]
    beats: list[dict[str, Any]]
    downbeats: list[dict[str, Any]]
    accents: list[dict[str, Any]]
    phrases: list[dict[str, Any]]
    sections: list[dict[str, Any]]
    confidence: dict[str, Any]
    manual_review_required: bool
    analysis_mode: str
    override_source: str | None
    applied_override_keys: list[str]
    notes: list[str]
    generated_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MotionUnitLibrary:
    schema_version: int
    library_id: str
    source_roots: dict[str, str]
    units: list[dict[str, Any]]
    notes: list[str]
    generated_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChoreographyPlan:
    schema_version: int
    plan_id: str
    song_id: str
    library_id: str
    steps: list[dict[str, Any]]
    notes: list[str]
    generated_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetargetReport:
    schema_version: int
    plan_id: str
    avatar_id: str
    status: str
    backend: dict[str, Any]
    sequence_reports: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    generated_at_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReviewBundle:
    schema_version: int
    bundle_id: str
    song_id: str
    plan_id: str
    song_summary: dict[str, Any]
    library_summary: dict[str, Any]
    choreography_summary: dict[str, Any]
    retarget_summary: dict[str, Any]
    highlights: list[str]
    generated_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
