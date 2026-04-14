from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SharedRoots:
    music_root: Path
    motion_base_assets_root: Path
    unity_reference_root: Path


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    outputs_root: Path
    config_path: Path
    shared_roots: SharedRoots
    blender_path: Path | None
    mesh_template_fbx: Path | None


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_app_config(project_root: Path | None = None) -> AppConfig:
    root = (project_root or default_project_root()).resolve()
    config_path = root / "config" / "paths.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    shared_payload = dict(payload.get("sharedRoots", {}))
    shared_roots = SharedRoots(
        music_root=(root / shared_payload["musicRoot"]).resolve(),
        motion_base_assets_root=(root / shared_payload["motionBaseAssetsRoot"]).resolve(),
        unity_reference_root=(root / shared_payload["unityReferenceRoot"]).resolve(),
    )
    blender_path_raw = payload.get("tooling", {}).get("blenderPath")
    blender_path = None
    if blender_path_raw:
        candidate = Path(blender_path_raw)
        blender_path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    mesh_template_raw = payload.get("tooling", {}).get("meshTemplateFbx")
    mesh_template_fbx = None
    if mesh_template_raw:
        candidate = Path(mesh_template_raw)
        mesh_template_fbx = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    return AppConfig(
        project_root=root,
        outputs_root=(root / "outputs").resolve(),
        config_path=config_path,
        shared_roots=shared_roots,
        blender_path=blender_path,
        mesh_template_fbx=mesh_template_fbx,
    )
