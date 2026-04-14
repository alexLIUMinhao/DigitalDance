from __future__ import annotations

from pathlib import Path

from .config import AppConfig


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_input_path(config: AppConfig, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = (config.project_root / candidate).resolve()
    else:
        candidate = candidate.resolve()

    allowed_roots = [
        config.project_root,
        config.shared_roots.music_root,
        config.shared_roots.motion_base_assets_root,
        config.shared_roots.unity_reference_root,
    ]
    if any(_is_within(candidate, root) for root in allowed_roots):
        return candidate

    raise ValueError(f"Input path is outside allowed roots: {candidate}")


def ensure_output_path(config: AppConfig, relative_output: str) -> Path:
    raw_output = Path(relative_output)
    if raw_output.is_absolute():
        output_path = raw_output.resolve()
    elif raw_output.parts and raw_output.parts[0] == "outputs":
        output_path = (config.project_root / raw_output).resolve()
    else:
        output_path = (config.outputs_root / raw_output).resolve()
    if not _is_within(output_path, config.outputs_root):
        raise ValueError(f"Output path must stay inside outputs/: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path
