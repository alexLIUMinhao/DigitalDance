from __future__ import annotations

import importlib
import importlib.util
import pickle
import shutil
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from music_motion_lab.source_bvh_validation import (
    BRIDGE_BONE_ORDER_22,
    BVH_TO_TARGET_BASIS,
    build_pose_payload,
    compare_pose_payloads,
    compute_bone_vector_diagnostics,
    extract_bvh_joint_positions,
    remap_per_joint_error_names,
    summarize_values,
)


SMPL_COMPAT_BONE_ORDER_24 = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
]

JOINT_ERROR_GROUPS = {
    "root": ["pelvis"],
    "pelvis": ["left_hip", "right_hip"],
    "spine": ["spine1", "spine2", "spine3", "neck", "head", "left_collar", "right_collar"],
    "arm": ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"],
    "leg": ["left_knee", "right_knee", "left_ankle", "right_ankle", "left_foot", "right_foot"],
}


@dataclass(frozen=True)
class SmplCompatClip:
    trans: np.ndarray
    quats: np.ndarray
    axis_angle: np.ndarray
    feature_dims: int
    frame_count: int


def import_module_from_path(module_name: str, path: Path, *, extra_sys_path: Path | None = None) -> ModuleType:
    inserted_paths: list[str] = []
    if extra_sys_path is not None:
        sys.path.insert(0, str(extra_sys_path))
        inserted_paths.append(str(extra_sys_path))
    sys.path.insert(0, str(path.parent))
    inserted_paths.append(str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"unable to import module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for inserted in inserted_paths:
            if sys.path and sys.path[0] == inserted:
                sys.path.pop(0)
            elif inserted in sys.path:
                sys.path.remove(inserted)


def _install_optional_module_stub(name: str, attrs: dict[str, Any]) -> None:
    try:
        importlib.import_module(name)
        return
    except ModuleNotFoundError:
        pass
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module


def install_character_animation_tools_stubs() -> None:
    class EasyDict(dict):
        def __getattr__(self, key: str) -> Any:
            try:
                return self[key]
            except KeyError as exc:
                raise AttributeError(key) from exc

        def __setattr__(self, key: str, value: Any) -> None:
            self[key] = value

    class YamlLoader:
        pass

    def _unsupported_loader(*_args: Any, **_kwargs: Any) -> Any:
        raise ModuleNotFoundError("optional config loader is not installed in this runtime")

    _install_optional_module_stub("easydict", {"EasyDict": EasyDict})
    _install_optional_module_stub("toml", {"load": _unsupported_loader})
    _install_optional_module_stub("yaml", {"load": _unsupported_loader, "Loader": YamlLoader})


def prepare_character_animation_tools_repo(repo_root: Path) -> Path:
    if sys.version_info >= (3, 10):
        return repo_root

    compat_root = repo_root.parent / "CharacterAnimationTools_py39_compat"
    compat_root.parent.mkdir(parents=True, exist_ok=True)
    if compat_root.exists():
        shutil.rmtree(compat_root)
    shutil.copytree(repo_root, compat_root)

    skel_path = compat_root / "anim" / "skel.py"
    skel_text = skel_path.read_text(encoding="utf-8")
    skel_old = """def axis_to_vector(axis: str):
    match axis:
        case "x":
            return [1, 0, 0]
        case "y":
            return [0, 1, 0]
        case "z":
            return [0, 0, 1]
        case "-x":
            return [-1, 0, 0]
        case "-y":
            return [0, -1, 0]
        case "-z":
            return [0, 0, -1]
        case _:
            raise ValueError
"""
    skel_new = """def axis_to_vector(axis: str):
    if axis == "x":
        return [1, 0, 0]
    if axis == "y":
        return [0, 1, 0]
    if axis == "z":
        return [0, 0, 1]
    if axis == "-x":
        return [-1, 0, 0]
    if axis == "-y":
        return [0, -1, 0]
    if axis == "-z":
        return [0, 0, -1]
    raise ValueError
"""
    if skel_old not in skel_text:
        raise ValueError("unable to patch CharacterAnimationTools skel.py for Python 3.9 compatibility")
    skel_path.write_text(skel_text.replace(skel_old, skel_new), encoding="utf-8")

    smpl_path = compat_root / "anim" / "smpl.py"
    smpl_text = smpl_path.read_text(encoding="utf-8")
    smpl_old = """def load_model(model_path: Path, gender: str=None):
    if isinstance(model_path, str):
        model_path = Path(model_path)
    if model_path.suffix == "":
        model_path = model_path / gender / "model.npz"
    match model_path.suffix:
        case ".npz":
            model_dict = np.load(model_path, allow_pickle=True)
        case ".pkl":
            try:
                model_dict = pickle_load(model_path)
            except:
                model_dict = pickle_load(model_path, encoding="latin1")
        case _ :  
            ValueError("This file is not supported.")
    return model_dict
"""
    smpl_new = """def load_model(model_path: Path, gender: str=None):
    if isinstance(model_path, str):
        model_path = Path(model_path)
    if model_path.suffix == "":
        model_path = model_path / gender / "model.npz"
    if model_path.suffix == ".npz":
        model_dict = np.load(model_path, allow_pickle=True)
    elif model_path.suffix == ".pkl":
        try:
            model_dict = pickle_load(model_path)
        except Exception:
            model_dict = pickle_load(model_path, encoding="latin1")
    else:
        raise ValueError("This file is not supported.")
    return model_dict
"""
    if smpl_old not in smpl_text:
        raise ValueError("unable to patch CharacterAnimationTools smpl.py for Python 3.9 compatibility")
    smpl_path.write_text(smpl_text.replace(smpl_old, smpl_new), encoding="utf-8")
    return compat_root


def clear_character_animation_tools_modules() -> None:
    for module_name in list(sys.modules.keys()):
        if module_name == "anim" or module_name.startswith("anim.") or module_name == "util" or module_name.startswith("util."):
            sys.modules.pop(module_name, None)


def build_smpl_compat_quats(full_quaternions: np.ndarray, convert_module: ModuleType) -> np.ndarray:
    full_payload = np.asarray(full_quaternions, dtype=np.float64)
    frame_count = int(full_payload.shape[0])
    compat = np.zeros((frame_count, len(SMPL_COMPAT_BONE_ORDER_24), 4), dtype=np.float64)
    compat[..., 3] = 1.0
    for target_index, joint_name in enumerate(SMPL_COMPAT_BONE_ORDER_24):
        source_name = convert_module.FINEDANCE_BRIDGE_JOINT_MAP[joint_name]
        source_index = convert_module.FINEDANCE_JOINT_INDEX[source_name]
        compat[:, target_index, :] = full_payload[:, source_index, :]
    return np.asarray(convert_module.safe_quaternions(compat), dtype=np.float64)


def load_finedance_smpl_compat_clip(raw_motion_path: Path, convert_module: ModuleType) -> SmplCompatClip:
    trans, full_quats, feature_dims = convert_module.load_finedance_motion_components(raw_motion_path)
    smpl_quats = build_smpl_compat_quats(full_quats, convert_module)
    axis_angle = Rotation.from_quat(smpl_quats.reshape((-1, 4))).as_rotvec().reshape((-1, len(SMPL_COMPAT_BONE_ORDER_24), 3))
    return SmplCompatClip(
        trans=np.asarray(trans, dtype=np.float64),
        quats=smpl_quats,
        axis_angle=np.asarray(axis_angle, dtype=np.float64),
        feature_dims=int(feature_dims),
        frame_count=int(trans.shape[0]),
    )


def stabilize_zero_rotvecs(axis_angle: np.ndarray, *, epsilon: float = 1e-8) -> np.ndarray:
    payload = np.asarray(axis_angle, dtype=np.float64).copy()
    norms = np.linalg.norm(payload, axis=-1)
    mask = norms <= float(epsilon)
    if np.any(mask):
        payload[mask, 0] = float(epsilon)
        payload[mask, 1] = 0.0
        payload[mask, 2] = 0.0
    return payload


def write_smpl_compat_npz(path: Path, clip: SmplCompatClip) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stabilized_axis_angle = stabilize_zero_rotvecs(clip.axis_angle)
    np.savez(
        path,
        poses=stabilized_axis_angle[np.newaxis, ...],
        trans=clip.trans[np.newaxis, ...],
    )


def write_aist_compat_pkl(path: Path, clip: SmplCompatClip, *, scaling: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "smpl_poses": clip.axis_angle.reshape((clip.frame_count, -1)),
        "smpl_scaling": float(scaling),
        "smpl_trans": clip.trans.copy(),
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def aligned_positions_for_frame_indices(all_positions: np.ndarray, frame_indices: list[int]) -> tuple[list[int], np.ndarray]:
    payload = np.asarray(all_positions, dtype=np.float64)
    if not frame_indices:
        return list(range(int(payload.shape[0]))), payload
    max_requested = max(frame_indices)
    if max_requested < int(payload.shape[0]):
        return list(frame_indices), payload[np.asarray(frame_indices, dtype=np.int64)]
    if int(payload.shape[0]) == len(frame_indices):
        return list(frame_indices), payload
    raise ValueError(
        f"requested frameIndices exceed candidate frame count: max_requested={max_requested} frame_count={int(payload.shape[0])}"
    )


def build_canonical_payload_from_bvh(
    *,
    input_path: Path,
    parse_bvh,
    convert_module: ModuleType,
    frame_indices: list[int],
    unit_scale: float = 1.0,
    basis_matrix: np.ndarray | None = BVH_TO_TARGET_BASIS,
) -> tuple[dict[str, Any], float, list[str]]:
    all_names, _all_frame_indices, all_positions, frame_time = extract_bvh_joint_positions(
        input_path,
        parse_bvh,
        bone_order=None,
        basis_matrix=basis_matrix,
    )
    index_map = convert_module.bone_index_map(all_names)
    selected_positions = np.stack(
        [all_positions[:, index_map[bone_name], :] for bone_name in BRIDGE_BONE_ORDER_22],
        axis=1,
    )
    if unit_scale != 1.0:
        selected_positions = selected_positions * float(unit_scale)
    aligned_frame_indices, aligned_positions = aligned_positions_for_frame_indices(selected_positions, frame_indices)
    payload = build_pose_payload(
        input_path=str(input_path),
        bone_order=list(BRIDGE_BONE_ORDER_22),
        frame_indices=aligned_frame_indices,
        poses=aligned_positions,
        include_preview_space=True,
        extra={
            "bvhFrameTime": float(frame_time),
            "sourceNodeNames": list(all_names),
            "unitScaleApplied": float(unit_scale),
        },
    )
    return payload, float(frame_time), list(all_names)


def compute_group_error_summary(
    truth_positions: np.ndarray,
    candidate_positions: np.ndarray,
    bone_order: list[str],
) -> dict[str, Any]:
    truth = np.asarray(truth_positions, dtype=np.float64).copy()
    candidate = np.asarray(candidate_positions, dtype=np.float64).copy()
    truth -= truth[:, :1, :]
    candidate -= candidate[:, :1, :]
    error = np.linalg.norm(candidate - truth, axis=-1)
    index_map = {name: index for index, name in enumerate(bone_order)}
    summary: dict[str, Any] = {}
    for group_name, bone_names in JOINT_ERROR_GROUPS.items():
        indices = [index_map[name] for name in bone_names if name in index_map]
        if not indices:
            continue
        stats = summarize_values(error[:, indices])
        stats["bones"] = [bone_order[index] for index in indices]
        summary[group_name] = stats
    return summary


def build_validation_result(
    *,
    route_name: str,
    official_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
    convert_module: ModuleType,
    bvh_path: Path,
    fk_joints_path: Path,
    frame_time: float,
    route_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = compare_pose_payloads(official_payload, candidate_payload, convert_module.compare_bridge_positions)
    bone_order = list(candidate_payload.get("boneOrder", []))
    per_joint = remap_per_joint_error_names(summary, bone_order)
    bone_diagnostics = compute_bone_vector_diagnostics(official_payload, candidate_payload)
    group_errors = compute_group_error_summary(
        np.asarray(official_payload["poses"], dtype=np.float64),
        np.asarray(candidate_payload["poses"], dtype=np.float64),
        bone_order,
    )
    result = {
        "routeName": route_name,
        "bvhPath": str(bvh_path),
        "fkJointsPath": str(fk_joints_path),
        "frameCountCompared": int(summary.get("frameCountCompared", 0) or 0),
        "jointCountCompared": int(summary.get("jointCountCompared", 0) or 0),
        "frameTime": float(frame_time),
        "jointPositionError": dict(summary.get("jointPositionError", {}) or {}),
        "perJointPositionError": per_joint,
        "groupPositionError": group_errors,
        "boneVectorDiagnostics": bone_diagnostics,
        "meta": dict(route_meta or {}),
    }
    return result


def run_external_smpl2bvh(
    *,
    repo_root: Path,
    poses_path: Path,
    output_bvh: Path,
    model_path: Path,
    fps: int = 30,
    model_type: str = "smplx",
    gender: str = "NEUTRAL",
) -> None:
    module = import_module_from_path(
        "external_smpl2bvh_module",
        repo_root / "smpl2bvh.py",
        extra_sys_path=repo_root,
    )
    output_bvh.parent.mkdir(parents=True, exist_ok=True)
    module.smpl2bvh(
        model_path=str(model_path),
        poses=str(poses_path),
        output=str(output_bvh),
        mirror=False,
        model_type=str(model_type),
        gender=str(gender),
        fps=int(fps),
    )


def run_external_character_animation_tools(
    *,
    repo_root: Path,
    poses_path: Path,
    smpl_path: Path,
    output_bvh: Path,
    fps: int = 30,
    scale: float = 1.0,
) -> None:
    install_character_animation_tools_stubs()
    compatible_repo_root = prepare_character_animation_tools_repo(repo_root)
    clear_character_animation_tools_modules()
    sys.path.insert(0, str(compatible_repo_root))
    try:
        aistpp = importlib.import_module("anim.aistpp")
        bvh = importlib.import_module("anim.bvh")
        anim = aistpp.load(
            aistpp_motion_path=poses_path,
            smpl_path=smpl_path,
            scale=float(scale),
            fps=int(fps),
        )
        output_bvh.parent.mkdir(parents=True, exist_ok=True)
        bvh.save(filepath=output_bvh, anim=anim)
    finally:
        if sys.path and sys.path[0] == str(compatible_repo_root):
            sys.path.pop(0)
