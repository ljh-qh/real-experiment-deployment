from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class UrdfVisual:
    link_name: str
    visual_index: int
    mesh_path: Path
    base_from_visual: np.ndarray
    scale: tuple[float, float, float]


@dataclass(frozen=True)
class UrdfScene:
    urdf_path: Path
    base_link: str
    link_transforms: dict[str, np.ndarray]
    visuals: tuple[UrdfVisual, ...]


def load_urdf_scene(
    urdf_path: str | Path,
    base_link: str,
    joint_positions_rad: dict[str, float] | None = None,
    package_paths: dict[str, str | Path] | None = None,
) -> UrdfScene:
    path = Path(urdf_path)
    root = ET.parse(path).getroot()
    link_names = _link_names(root)
    if base_link not in link_names:
        raise ValueError(f"URDF base_link={base_link} not found")

    link_transforms = _compute_link_transforms(root, base_link, joint_positions_rad)
    packages = {} if package_paths is None else {name: Path(value) for name, value in package_paths.items()}
    visuals = _load_visuals(root, path, link_transforms, packages)
    return UrdfScene(
        urdf_path=path,
        base_link=base_link,
        link_transforms=link_transforms,
        visuals=tuple(visuals),
    )


def _link_names(root: ET.Element) -> set[str]:
    return {link.attrib["name"] for link in root.findall("link")}


def _compute_link_transforms(
    root: ET.Element,
    base_link: str,
    joint_positions_rad: dict[str, float] | None,
) -> dict[str, np.ndarray]:
    children_by_parent: dict[str, list[tuple[str, np.ndarray]]] = {}
    joint_values = {} if joint_positions_rad is None else joint_positions_rad
    for joint in root.findall("joint"):
        parent_link = joint.find("parent").attrib["link"]
        child_link = joint.find("child").attrib["link"]
        parent_from_child = _origin_matrix(joint.find("origin")) @ _joint_motion_matrix(joint, joint_values)
        if parent_link not in children_by_parent:
            children_by_parent[parent_link] = []
        children_by_parent[parent_link].append((child_link, parent_from_child))

    link_transforms = {base_link: np.eye(4, dtype=np.float64)}
    pending = [base_link]
    while pending:
        parent_link = pending.pop(0)
        if parent_link not in children_by_parent:
            continue
        base_from_parent = link_transforms[parent_link]
        for child_link, parent_from_child in children_by_parent[parent_link]:
            if child_link in link_transforms:
                raise ValueError(f"URDF link={child_link} reached more than once")
            link_transforms[child_link] = base_from_parent @ parent_from_child
            pending.append(child_link)
    return link_transforms


def _joint_motion_matrix(joint: ET.Element, joint_positions_rad: dict[str, float]) -> np.ndarray:
    joint_type = joint.attrib["type"]
    if joint_type == "fixed":
        return np.eye(4, dtype=np.float64)
    if joint_type in ("revolute", "continuous"):
        q_rad = _joint_value_rad(joint, joint_positions_rad)
        axis = np.asarray(_joint_axis(joint), dtype=np.float64)
        motion = np.eye(4, dtype=np.float64)
        motion[:3, :3] = Rotation.from_rotvec(axis * q_rad).as_matrix()
        return motion
    if joint_type == "prismatic":
        q_m = _joint_value_rad(joint, joint_positions_rad)
        axis = np.asarray(_joint_axis(joint), dtype=np.float64)
        motion = np.eye(4, dtype=np.float64)
        motion[:3, 3] = axis * q_m
        return motion
    raise ValueError(f"unsupported URDF joint type: {joint_type}")


def _joint_value_rad(joint: ET.Element, joint_positions_rad: dict[str, float]) -> float:
    joint_name = joint.attrib["name"]
    if joint_name in joint_positions_rad:
        return float(joint_positions_rad[joint_name])
    return 0.0


def _joint_axis(joint: ET.Element) -> tuple[float, float, float]:
    axis = joint.find("axis")
    if axis is None:
        raise ValueError(f"URDF joint={joint.attrib['name']} missing axis")
    values = np.asarray(_parse_vector3(axis.attrib["xyz"]), dtype=np.float64)
    norm = float(np.linalg.norm(values))
    if norm <= 0.0:
        raise ValueError(f"URDF joint={joint.attrib['name']} axis must be non-zero")
    normalized = values / norm
    return (float(normalized[0]), float(normalized[1]), float(normalized[2]))


def _load_visuals(
    root: ET.Element,
    urdf_path: Path,
    link_transforms: dict[str, np.ndarray],
    package_paths: dict[str, Path],
) -> list[UrdfVisual]:
    visuals = []
    for link in root.findall("link"):
        link_name = link.attrib["name"]
        if link_name not in link_transforms:
            continue
        base_from_link = link_transforms[link_name]
        for visual_index, visual in enumerate(link.findall("visual")):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            mesh_path = _resolve_mesh_path(urdf_path.parent, mesh.attrib["filename"], package_paths)
            if not mesh_path.exists():
                raise FileNotFoundError(str(mesh_path))
            link_from_visual = _origin_matrix(visual.find("origin"))
            visuals.append(
                UrdfVisual(
                    link_name=link_name,
                    visual_index=visual_index,
                    mesh_path=mesh_path,
                    base_from_visual=base_from_link @ link_from_visual,
                    scale=_mesh_scale(mesh),
                )
            )
    return visuals


def _origin_matrix(origin: ET.Element | None) -> np.ndarray:
    xyz = (0.0, 0.0, 0.0)
    rpy = (0.0, 0.0, 0.0)
    if origin is not None:
        if "xyz" in origin.attrib:
            xyz = _parse_vector3(origin.attrib["xyz"])
        if "rpy" in origin.attrib:
            rpy = _parse_vector3(origin.attrib["rpy"])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    transform[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return transform


def _mesh_scale(mesh: ET.Element) -> tuple[float, float, float]:
    if "scale" not in mesh.attrib:
        return (1.0, 1.0, 1.0)
    return _parse_vector3(mesh.attrib["scale"])


def _parse_vector3(text: str) -> tuple[float, float, float]:
    values = tuple(float(value) for value in text.split())
    if len(values) != 3:
        raise ValueError(f"expected 3 values, got {text}")
    return (values[0], values[1], values[2])


def _resolve_mesh_path(urdf_dir: Path, filename: str, package_paths: dict[str, Path]) -> Path:
    if filename.startswith("package://"):
        package_relative_path = filename.removeprefix("package://")
        package_name, relative_path = package_relative_path.split("/", maxsplit=1)
        if package_name not in package_paths:
            raise ValueError(f"URDF package={package_name} not found for mesh path: {filename}")
        return (package_paths[package_name] / relative_path).resolve()
    mesh_path = Path(filename)
    if mesh_path.is_absolute():
        return mesh_path
    return (urdf_dir / mesh_path).resolve()
