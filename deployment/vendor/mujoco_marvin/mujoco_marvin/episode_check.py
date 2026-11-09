"""Offline episode audit. No vendor SDK is imported and no hardware is opened."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


class EpisodeError(ValueError):
    pass


def timestamps(values, label, errors, max_gap=.25):
    values = np.asarray(values)
    if values.ndim != 1 or len(values) < 2:
        errors.append(f"{label}: fewer than two timestamped samples")
        return None
    if not np.issubdtype(values.dtype, np.integer):
        errors.append(f"{label}: sys_time_ns must be integer nanoseconds")
        return None
    delta = np.diff(values)
    if np.any(values <= 0) or np.any(delta <= 0):
        errors.append(f"{label}: timestamps are nonpositive, duplicated or out of order")
        return None
    gap = float(delta.max()) / 1e9
    if gap > max_gap:
        errors.append(f"{label}: maximum sample gap {gap:.3f}s exceeds {max_gap:.3f}s")
    return {"samples": len(values), "start_mono": int(values[0]) / 1e9,
            "stop_mono": int(values[-1]) / 1e9, "max_gap_s": gap,
            "mean_hz": float((len(values) - 1) * 1e9 / (values[-1] - values[0]))}


def matrix(group, name, width, rows, errors):
    if name not in group:
        errors.append(f"{group.name}/{name}: missing dataset")
        return None
    values = np.asarray(group[name])
    real_numeric = (np.issubdtype(values.dtype, np.integer)
                    or np.issubdtype(values.dtype, np.floating))
    if values.shape != (rows, width) or not real_numeric:
        errors.append(f"{group.name}/{name}: expected ({rows}, {width}), got {values.shape}")
        return None
    if not np.all(np.isfinite(values)):
        errors.append(f"{group.name}/{name}: contains NaN/Inf")
        return None
    return values


def coverage(stats, start, stop, label, errors, slack):
    if stats is None:
        return
    if stats["start_mono"] > start + slack or stats["stop_mono"] < stop - slack:
        errors.append(f"{label}: stream does not cover episode boundaries within {slack:.3f}s")
    if stats["start_mono"] < start - slack or stats["stop_mono"] > stop + slack:
        errors.append(f"{label}: samples outside episode time range")


def inspect_episode(path, *, devices="both", cameras=("wrist", "main", "wrist2"),
                    require_tactile=True, decode_video=True, max_gap=.25, slack=.5):
    """Return explicit errors rather than treating replayability as completeness."""
    path = Path(path)
    errors, warnings, streams = [], [], {}
    report = {"episode": str(path.resolve()), "ok": False, "errors": errors,
              "warnings": warnings, "streams": streams, "cameras": {},
              "scope": {"devices": devices, "cameras": list(cameras),
                        "tactile_required": require_tactile, "video_decoded": decode_video,
                        "max_joint_rgb_gap_s": max_gap, "edge_slack_s": slack},
              "known_unrecorded": ["MANUS raw skeleton",
                                   "camera calibration", "task metadata/calibration snapshots"]}
    try:
        manifest = json.loads((path / "manifest.json").read_text())
        start, stop = float(manifest["start_mono"]), float(manifest["stop_mono"])
        if not np.isfinite([start, stop]).all() or not 0 < start < stop:
            raise ValueError("invalid start_mono/stop_mono")
        if manifest.get("status") == "incomplete" or manifest.get("recording_errors"):
            errors.append(f"Recorder marked episode incomplete: {manifest.get('recording_errors', [])}")
        report["duration_s"] = stop - start
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f"manifest.json: {exc}")
        return report

    try:
        with h5py.File(path / "robot_states.h5", "r") as h5:
            for name, count_key, optional in (("pico_teleop", "robot_rows", False),
                                              ("sharpa_joints", "sharpa_joint_rows", False),
                                              ("sharpa_commands", "sharpa_command_rows", True)):
                if devices == "hands" and name == "pico_teleop":
                    continue
                if devices == "arms" and name in ("sharpa_joints", "sharpa_commands"):
                    continue
                key = "robot_state/" + name
                if key not in h5 or "sys_time_ns" not in h5[key]:
                    if not optional:
                        errors.append(f"{key}: missing required joint stream")
                    continue
                group = h5[key]
                times = np.asarray(group["sys_time_ns"])
                rows = len(times) if times.ndim == 1 else 0
                stat = timestamps(times, key, errors, max_gap)
                streams[name] = stat
                coverage(stat, start, stop, key, errors, slack)
                if manifest.get(count_key) != rows:
                    errors.append(f"{count_key}: manifest {manifest.get(count_key)} != HDF5 {rows}")
                if name == "pico_teleop":
                    fields = {"command_rad": 14, "real_left_deg": 7, "real_right_deg": 7}
                    if devices != "arms":
                        # Tracker poses are inputs, unused by arms-only replay;
                        # occlusion NaNs there must not reject the episode.
                        fields.update({"pose_" + r: 7 for r in (
                            "left_wrist", "right_wrist", "left_arm", "right_arm")})
                    # real_tcp_* only exists in episodes recorded with FK logging;
                    # validate when present, stay silent for legacy files.
                    for side in ("left", "right"):
                        for suffix, width in (("pos", 3), ("quat", 4)):
                            name = f"real_tcp_{side}_{suffix}"
                            if name in group:
                                fields[name] = width
                else:
                    fields = {"left_rad": 22, "right_rad": 22}
                    units = group.attrs.get("units")
                    if isinstance(units, bytes):
                        units = units.decode()
                    report[f"{name}_schema"] = {
                        "units": units.decode() if isinstance(units, bytes) else units,
                        "native_joint_order": False,
                    }
                    if units is not None and units != "rad":
                        errors.append(f"{name}: unexpected units")
                    if units is None:
                        warnings.append(f"{name}: units/order metadata absent; legacy schema assumes SDK radians")
                    from .sharpa_backend import JOINT_NAMES
                    if "joint_names" in group.attrs:
                        names = tuple(n.decode() if isinstance(n, bytes) else n
                                      for n in group.attrs["joint_names"])
                        if names != JOINT_NAMES:
                            errors.append(f"{name}: joint order differs from native SDK order")
                        else:
                            report[f"{name}_schema"]["native_joint_order"] = True
                    else:
                        warnings.append(f"{name}: joint-order metadata absent; hardware replay will refuse this file")
                    if name == "sharpa_joints":
                        for side in ("left", "right"):
                            sn_key = side + "_sn"
                            if sn_key in group:
                                if group[sn_key].shape != (rows,):
                                    errors.append(f"sharpa_joints/{sn_key}: inconsistent identity rows")
                                    continue
                                serials = set(group[sn_key].asstr()[:])
                                if len(serials) != 1 or next(iter(serials), "") in ("", "unknown"):
                                    errors.append(f"sharpa_joints/{sn_key}: missing or changing device identity")
                                else:
                                    report[sn_key] = next(iter(serials))
                            else:
                                warnings.append(f"{sn_key} absent from legacy joint stream")
                for field, width in fields.items():
                    values = matrix(group, field, width, rows, errors)
                    if values is not None and len(values):
                        if field.startswith("pose_"):
                            if np.any(np.abs(np.linalg.norm(values[:, 3:], axis=1) - 1.) > .02):
                                errors.append(f"{field}: invalid tracker quaternion")
                        elif field.endswith("_quat"):
                            if np.any(np.abs(np.linalg.norm(values, axis=1) - 1.) > .02):
                                errors.append(f"{field}: invalid quaternion")
                        elif field.endswith("_rad"):
                            if np.any(np.abs(values) > 2 * np.pi):
                                errors.append(f"{field}: implausible radians (possible degree/radian mismatch)")
                            if np.all(np.ptp(values, axis=0) < 1e-6):
                                warnings.append(f"{field}: constant throughout; verify the trial included motion")
            total_tactile = 0
            root = h5.get("robot_state/sharpa_tactile")
            tactile_h5 = None
            if root is None and (path / "tactile.h5").is_file():
                # 新架构：触觉由独立子进程写 tactile.h5（顶层 sharpa_tactile/<key>）
                tactile_h5 = h5py.File(path / "tactile.h5", "r")
                root = tactile_h5.get("sharpa_tactile")
            sides = set()
            tactile_times = {"left": [], "right": []}
            if root is not None:
                for key, group in root.items():
                    if not isinstance(group, h5py.Group) or "sys_time_ns" not in group:
                        errors.append(f"sharpa_tactile/{key}: malformed group")
                        continue
                    times = np.asarray(group["sys_time_ns"])
                    rows = len(times)
                    total_tactile += rows
                    stat = timestamps(times, "tactile/" + key, errors, max_gap=max(2., max_gap))
                    # Individual fingers may be sparse; report coverage without asserting continuous contact.
                    streams["tactile/" + key] = stat
                    matrix(group, "force", 3, rows, errors)
                    declared_tactile = set(manifest.get("tactile_streams", []))
                    if "torque" in declared_tactile:
                        matrix(group, "torque", 3, rows, errors)
                    if "depth" in declared_tactile:
                        depth = group.get("depth")
                        if (depth is None or depth.ndim not in (3, 4)
                                or depth.shape[0] != rows or 0 in depth.shape[1:]):
                            errors.append(f"tactile/{key}/depth: missing or inconsistent rows")
                    if "contact_point" in declared_tactile:
                        contact = group.get("contact_point")
                        if (contact is None or contact.ndim != 2 or contact.shape[0] != rows
                                or contact.shape[1] % 3 != 0 or contact.shape[1] == 0):
                            errors.append(f"tactile/{key}/contact_point: missing or inconsistent rows")
                    if "dist_force" in declared_tactile:
                        dist = group.get("dist_force")
                        if (dist is None or dist.ndim != 4 or dist.shape[0] != rows
                                or dist.shape[1:] != (60, 60, 3)):
                            errors.append(f"tactile/{key}/dist_force: missing or inconsistent rows")
                    if "raw" in declared_tactile:
                        raw = group.get("raw")
                        if (raw is None or raw.ndim != 3 or raw.shape[0] != rows
                                or 0 in raw.shape[1:]):
                            errors.append(f"tactile/{key}/raw: missing or inconsistent rows")
                    if "frame_id" in declared_tactile:
                        fid = group.get("frame_id")
                        if fid is None or fid.shape != (rows,):
                            errors.append(f"tactile/{key}/frame_id: missing or inconsistent rows")
                        elif np.any(np.diff(np.asarray(fid)) <= 0):
                            warnings.append(f"tactile/{key}/frame_id: not strictly increasing")
                    for name in ("side", "serial", "finger"):
                        if name not in group or group[name].shape != (rows,):
                            errors.append(f"tactile/{key}/{name}: missing or inconsistent rows")
                    if "side" in group:
                        labels = group["side"].asstr()[:]
                        sides.update(labels)
                        if labels.shape == times.shape:
                            for side in tactile_times:
                                tactile_times[side].extend(times[labels == side].tolist())
                    if "serial" in group and any(s in ("", "unknown") for s in group["serial"].asstr()[:]):
                        errors.append(f"tactile/{key}: missing device serial")
            if manifest.get("tactile_rows", 0) != total_tactile:
                errors.append("tactile_rows: manifest/HDF5 mismatch")
            if require_tactile and (total_tactile < 2 or sides != {"left", "right"}):
                errors.append("Tactile data from both hands is required")
            if require_tactile:
                for side, values in tactile_times.items():
                    stat = timestamps(np.unique(values).astype(np.int64), "tactile/" + side,
                                      errors, max_gap=max(2., max_gap))
                    coverage(stat, start, stop, "tactile/" + side, errors, slack)
            if tactile_h5 is not None:
                tactile_h5.close()
    except (OSError, ValueError, TypeError, KeyError) as exc:
        errors.append(f"robot_states.h5: {exc}")

    declared = manifest.get("cameras", {})
    if devices == "arms":
        # Arms-only replay consumes no video; stripped/absent cameras are fine.
        declared, cameras = {}, ()
    for role in dict.fromkeys([*cameras, *declared]):
        try:
            report["cameras"][role] = inspect_camera(path, role, declared.get(role),
                                                     start, stop, errors, decode_video,
                                                     max_gap=max_gap, slack=slack)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            errors.append(f"camera {role}: {exc}")
    report["ok"] = not errors
    return report


def inspect_camera(episode, role, declared, start, stop, errors, decode, max_gap, slack):
    if not declared:
        raise ValueError("missing camera in manifest")
    folder = episode / role
    if folder.resolve().parent != episode.resolve():
        raise ValueError("invalid camera role/path")
    indexes = sorted(folder.glob("timestamps_take*.jsonl"))
    if not indexes:
        raise ValueError("missing timestamp index")
    all_times, depth_paths, depth_times, frame_numbers = [], [], [], []
    count, video_count = 0, 0
    from PIL import Image
    for index in indexes:
        rows = [json.loads(line) for line in index.read_text().splitlines() if line.strip()]
        if not rows or rows[0].get("event") != "start" or rows[-1].get("event") != "stop":
            errors.append(f"{role}/{index.name}: segment not finalized")
        frames = [r for r in rows if "i" in r]
        count += len(frames)
        frame_numbers.extend(r["i"] for r in frames)
        all_times.extend(float(r["mono"]) for r in frames)
        for row in frames:
            if row.get("depth"):
                depth = folder / row["depth"]
                if not depth.resolve().is_relative_to((folder / "depth").resolve()):
                    raise ValueError("depth path escapes its camera folder")
                with Image.open(depth) as img:
                    if img.format != "PNG" or img.mode not in ("I;16", "I;16L", "I;16B", "I"):
                        raise ValueError("depth is not 16-bit PNG")
                    img.load()
                    values = np.asarray(img)
                    if values.ndim != 2 or values.min() < 0 or values.max() > 65535:
                        raise ValueError("invalid depth pixels")
                depth_paths.append(depth.resolve())
                depth_times.append(float(row["mono"]))
        video = folder / index.name.replace("timestamps_", "color_").replace(".jsonl", ".mp4")
        if not video.is_file() or video.stat().st_size == 0:
            raise ValueError(f"missing/empty video {video.name}")
        if decode:
            import imageio_ffmpeg
            nframes, _ = imageio_ffmpeg.count_frames_and_secs(str(video))
            video_count += nframes
            if nframes != len(frames):
                errors.append(f"{role}: decoded video frames {nframes} != timestamps {len(frames)}")
    stat = timestamps(np.rint(np.asarray(all_times) * 1e9).astype(np.int64), role, errors, max_gap)
    coverage(stat, start, stop, role, errors, slack)
    depth_stat = timestamps(np.rint(np.asarray(depth_times) * 1e9).astype(np.int64),
                            role + "/depth", errors, max_gap=max(.5, max_gap))
    coverage(depth_stat, start, stop, role + "/depth", errors, slack)
    if frame_numbers != list(range(1, count + 1)):
        errors.append(f"{role}: missing/duplicate/out-of-order frame numbers")
    disk_depth = set(p.resolve() for p in (folder / "depth").glob("*.png"))
    if len(depth_paths) != len(set(depth_paths)) or set(depth_paths) != disk_depth:
        errors.append(f"{role}: depth index/file mismatch or duplicate references")
    if not depth_paths or declared.get("depth_frames") != len(depth_paths):
        errors.append(f"{role}: missing depth or manifest count mismatch")
    if declared.get("frames") != count:
        errors.append(f"{role}: RGB frame count disagrees with manifest")
    return {"frames": count, "depth_frames": len(depth_paths),
            "decoded_video_frames": video_count if decode else None, "timing": stat,
            "depth_timing": depth_stat}


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
