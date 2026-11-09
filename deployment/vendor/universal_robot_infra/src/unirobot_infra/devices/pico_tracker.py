"""PICO tracker acquisition and live-pose state shared by every teleop backend."""

import math
import os
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from unirobot_infra.calibration.tracker_alignment import rotation_matrix
from unirobot_infra.robots.arm import marvin_model


ROLES = ("left_wrist", "right_wrist", "left_arm", "right_arm")


def default_alignment_path():
    """Saved direction calibration; ``MUJOCO_MARVIN_CALIBRATION`` or
    ``./.local/pico-axis-calibration.json`` (created on demand)."""
    return Path(os.environ.get("MUJOCO_MARVIN_CALIBRATION")
                or Path.cwd() / ".local" / "pico-axis-calibration.json")


class TrackerActivity:
    """Report position and orientation activity separately from packet arrival.

    Rotation can keep updating while the source repeats its last position.
    Neither a new packet nor a changed quaternion proves live XYZ tracking.
    These are activity observations, not an SDK tracking-validity flag.
    """

    def __init__(self, serials):
        self.serials = dict(serials)
        self.last_frame = None
        self.samples = {}

    def observe(self, poses, frame, now=None):
        if not frame or frame == self.last_frame:
            return
        now = time.monotonic() if now is None else now
        self.last_frame = frame
        for role, serial in self.serials.items():
            pose = poses.get(serial)
            if pose is None:
                continue
            pose = np.asarray(pose, dtype=float)
            if pose.shape != (7,) or not np.isfinite(pose).all():
                continue
            norm = np.linalg.norm(pose[3:])
            if not .95 <= norm <= 1.05:
                continue
            pose = pose.copy()
            pose[3:] /= norm
            previous = self.samples.get(role)
            # Compare against the last moving anchor so gradual real motion
            # accumulates. Ignore sub-0.1 mm float noise and quaternion signs.
            position_changed = (previous is None or
                                np.linalg.norm(pose[:3] - previous["position_anchor"]) > .0001)
            rotation_changed = (previous is None or
                                abs(np.dot(pose[3:], previous["rotation_anchor"])) <
                                np.cos(np.deg2rad(.1) / 2.))
            self.samples[role] = {
                "pose": pose.copy(), "seen_at": now,
                "position_anchor": pose[:3].copy() if position_changed else previous["position_anchor"],
                "rotation_anchor": pose[3:].copy() if rotation_changed else previous["rotation_anchor"],
                "position_changed_at": now if position_changed else previous["position_changed_at"],
                "rotation_changed_at": now if rotation_changed else previous["rotation_changed_at"],
            }

    def component_status(self, role, component, now=None):
        if component not in ("position", "rotation"):
            raise ValueError("component must be position or rotation")
        now = time.monotonic() if now is None else now
        sample = self.samples.get(role)
        if sample is None:
            return "NO DATA", None
        if now - sample["seen_at"] > .25:
            return "NO FRESH PACKETS", now - sample["seen_at"]
        age = max(0., now - sample[f"{component}_changed_at"])
        return ("UNCHANGED" if age >= 2. else "UPDATING"), age

    def status(self, role, now=None):
        position, age = self.component_status(role, "position", now)
        rotation, rotation_age = self.component_status(role, "rotation", now)
        if position == "UNCHANGED" and rotation == "UPDATING":
            return "XYZ UNCHANGED / ROT UPDATING", age
        if position == rotation == "UNCHANGED":
            return "UNCHANGED", min(age, rotation_age)
        return position, age

    def diagnostic_lines(self, now=None):
        now = time.monotonic() if now is None else now
        lines = []
        for role, serial in self.serials.items():
            components = []
            for label, component in (("XYZ", "position"), ("ROT", "rotation")):
                state, age = self.component_status(role, component, now)
                suffix = f" {age:.1f}s" if age is not None and state != "UPDATING" else ""
                components.append(f"{label}: {state}{suffix}")
            lines.append(f"{role} [{serial[-5:]}]: " + " | ".join(components))
        if any(self.component_status(role, "position", now)[0] == "UNCHANGED" for role in self.serials):
            lines.append("XYZ UNCHANGED: stationary or position tracking held. Rotation alone does not prove live XYZ.")
        return lines


class PoseFreezeGuard:
    """Hold poses whose source stops updating, then ramp back on recovery.

    Live optical tracking never repeats bit-identical values at 90 Hz; a
    tracker that lost optical tracking repeats its last pose exactly.
    Components (position, rotation) are detected independently: while the
    source is stuck the frozen value is held, and on recovery the output
    blends toward the live value over ``ramp_seconds`` so downstream targets
    never teleport. Sub-threshold recoveries snap back instantly.
    """

    LIVE, HELD, RAMPING = "LIVE", "HELD", "RAMPING"

    def __init__(self, serials, freeze_frames=10, ramp_seconds=.6,
                 min_jump_m=.01, min_jump_deg=3.):
        if not isinstance(freeze_frames, int) or freeze_frames < 1:
            raise ValueError("freeze_frames must be a positive integer")
        if not math.isfinite(ramp_seconds) or ramp_seconds <= 0:
            raise ValueError("ramp_seconds must be positive and finite")
        self.roles = {serial: role for role, serial in serials.items()}
        self.freeze_frames = freeze_frames
        self.ramp_seconds = float(ramp_seconds)
        self.min_jump_m = float(min_jump_m)
        self.min_jump_rad = math.radians(min_jump_deg)
        self.last_frame = None
        self.state = {}
        self._last_poses = {}

    def modes(self):
        """{serial: (position_mode, rotation_mode)} — LIVE/HELD/RAMPING per component."""
        return {serial: (st["mode_pos"], st["mode_rot"])
                for serial, st in self.state.items()}

    @staticmethod
    def _close(a, b, is_position):
        if is_position:
            return bool(np.max(np.abs(a - b)) <= 1e-7)
        return abs(float(np.dot(a, b))) >= 1. - 1e-10

    def _component(self, st, key, live, is_position, now, transitions):
        """Advance one component (position or rotation) and return its output."""
        role, serial, component = st["role"], st["serial"], ("position" if is_position else "rotation")
        anchor_key, count_key = f"anchor_{key}", f"unchanged_{key}"
        mode_key, held_key = f"mode_{key}", f"held_{key}"
        changed = anchor_key not in st or not self._close(live, st[anchor_key], is_position)
        if changed:
            st[anchor_key] = live.copy()
            st[count_key] = 0
            if st[mode_key] == self.HELD:
                held = st[held_key]
                jump = (np.linalg.norm(live - held) if is_position else
                        2. * math.acos(min(1., abs(float(np.dot(live, held))))))
                limit = self.min_jump_m if is_position else self.min_jump_rad
                if jump < limit:
                    st[mode_key] = self.LIVE
                    transitions.append((role, serial, component, self.HELD, self.LIVE))
                else:
                    st[mode_key] = self.RAMPING
                    st[f"ramp_from_{key}"] = held.copy()
                    st[f"ramp_start_{key}"] = now
                    transitions.append((role, serial, component, self.HELD, self.RAMPING))
        else:
            st[count_key] += 1
            if st[count_key] >= self.freeze_frames and st[mode_key] != self.HELD:
                current = self._output(st, key, live, now)
                st[held_key] = current
                if st[mode_key] != self.HELD:
                    transitions.append((role, serial, component, st[mode_key], self.HELD))
                st[mode_key] = self.HELD
        return self._output(st, key, live, now)

    def _output(self, st, key, live, now):
        mode = st[f"mode_{key}"]
        if mode == self.LIVE:
            return live.copy()
        if mode == self.HELD:
            return st[f"held_{key}"].copy()
        progress = min(1., (now - st[f"ramp_start_{key}"]) / self.ramp_seconds)
        if progress >= 1.:
            st[f"mode_{key}"] = self.LIVE
            return live.copy()
        weight = progress * progress * (3. - 2. * progress)
        start = st[f"ramp_from_{key}"]
        if key == "pos":
            return (1. - weight) * start + weight * live
        sign = -1. if float(np.dot(start, live)) < 0. else 1.
        blended = (1. - weight) * start + weight * sign * live
        return blended / np.linalg.norm(blended)

    def filter(self, poses, frame, now=None):
        """Return (guarded poses, transitions) for one SDK frame.

        ``poses`` maps serial -> 7-float pose. Transitions are
        (role, serial, component, old_mode, new_mode) tuples for logging.
        """
        now = time.monotonic() if now is None else now
        transitions = []
        if not frame or frame == self.last_frame:
            if self.last_frame is None:
                return dict(poses), transitions
            return self._last_poses, transitions
        self.last_frame = frame
        out = {}
        for serial, role in self.roles.items():
            pose = poses.get(serial)
            st = self.state.get(serial)
            if st is None:
                st = {"role": role, "serial": serial,
                      "mode_pos": self.LIVE, "mode_rot": self.LIVE,
                      "unchanged_pos": 0, "unchanged_rot": 0}
                self.state[serial] = st
            if pose is None:
                # Missing packet: hold whatever this tracker last produced.
                if "held_pos" in st:
                    out[serial] = np.concatenate([st["held_pos"], st["held_rot"]])
                continue
            pose = np.asarray(pose, dtype=float)
            if pose.shape != (7,):
                continue
            pos = self._component(st, "pos", pose[:3], True, now, transitions)
            rot = self._component(st, "rot", pose[3:], False, now, transitions)
            out[serial] = np.concatenate([pos, rot])
        self._last_poses = out
        return out, transitions


def _sign_fix(quat, reference):
    return -quat if float(np.dot(quat, reference)) < 0. else quat


def _smoothstep(progress):
    return progress * progress * (3. - 2. * progress)


class ForearmRedundancy:
    """Fuse primary + aux trackers on the same forearm into wrist/elbow role poses.

    The two trackers ride ~180° apart on one forearm, so a wrist roll that
    turns one tracker's LEDs away from the headset leaves the other visible.
    While both are live, their relative transform is tracked with an EMA; when
    the primary freezes, the wrist pose is synthesized from the aux tracker
    through that transform, so output stays continuous instead of freezing.
    The elbow role is extrapolated from the forearm axis (wrist -> aux), with
    the axis remembered in the wrist frame as a fallback when the aux tracker
    is lost. Output is keyed by the downstream role serials, so the teleop
    core, direction calibration and freeze guard are unchanged.
    """

    def __init__(self, physical, out_serials, forearm_length=0., extra_length=.10,
                 switch_seconds=.3, ema_alpha=.05):
        """physical: {side: {"wrist": sn, "aux": sn}}; out_serials: ROLES dict
        whose ``*_arm`` entries are virtual placeholders. ``forearm_length`` <= 0
        measures the wrist->elbow length as tracker spacing + ``extra_length``."""
        if not math.isfinite(switch_seconds) or switch_seconds <= 0:
            raise ValueError("switch_seconds must be positive and finite")
        if not math.isfinite(ema_alpha) or not 0 < ema_alpha <= 1:
            raise ValueError("ema_alpha must be in (0, 1]")
        if not math.isfinite(forearm_length) or forearm_length < 0:
            raise ValueError("forearm_length must be nonnegative and finite")
        if not math.isfinite(extra_length) or extra_length < 0:
            raise ValueError("extra_length must be nonnegative and finite")
        self.out_serials = dict(out_serials)
        self.forearm_length = float(forearm_length)
        self.extra_length = float(extra_length)
        self.switch_seconds = float(switch_seconds)
        self.ema_alpha = float(ema_alpha)
        self.state = {}
        for side, pair in physical.items():
            self.state[side] = {"sn_wrist": pair["wrist"], "sn_aux": pair["aux"],
                                "source": None, "t_rel": None, "r_rel": None,
                                "axis_local": None, "length": None,
                                "switch_from": None, "switch_start": None,
                                "wrist": None, "elbow": None}
        self.sources = {side: None for side in physical}

    def _ema_vector(self, old, new):
        return new if old is None else (1. - self.ema_alpha) * old + self.ema_alpha * new

    def _ema_quat(self, old, new):
        if old is None:
            return new / np.linalg.norm(new)
        new = _sign_fix(new, old)
        blended = (1. - self.ema_alpha) * old + self.ema_alpha * new
        return blended / np.linalg.norm(blended)

    def _estimate(self, st, wrist, aux):
        """Track the aux->wrist transform and the wrist-frame forearm axis."""
        r_wrist, r_aux = Rotation.from_quat(wrist[3:]), Rotation.from_quat(aux[3:])
        st["t_rel"] = self._ema_vector(st["t_rel"], r_aux.inv().apply(wrist[:3] - aux[:3]))
        st["r_rel"] = self._ema_quat(st["r_rel"], (r_aux.inv() * r_wrist).as_quat())
        axis = aux[:3] - wrist[:3]
        spacing = np.linalg.norm(axis)
        if spacing > .05:
            st["axis_local"] = self._ema_vector(st["axis_local"],
                                                r_wrist.inv().apply(axis / spacing))
            if st["length"] is None:
                st["length"] = (self.forearm_length if self.forearm_length > 0
                                else spacing + self.extra_length)

    def _choose(self, st, poses, live):
        """Pick this side's wrist source; keep the relative transform fresh."""
        wrist, aux = poses.get(st["sn_wrist"]), poses.get(st["sn_aux"])
        live_wrist = live.get(st["sn_wrist"], False) and wrist is not None
        live_aux = live.get(st["sn_aux"], False) and aux is not None
        if live_wrist and live_aux:
            self._estimate(st, wrist, aux)
        if live_wrist:
            desired = "wrist"
        elif live_aux and st["t_rel"] is not None:
            desired = "aux"
        else:
            return None  # both lost: caller holds the last output
        if desired != st["source"] and st["wrist"] is not None:
            st["switch_from"] = st["wrist"].copy()
            st["switch_start"] = None  # armed; starts when the new pose lands
        elif desired != st["source"]:
            st["switch_from"] = None
        st["source"] = desired
        if desired == "wrist":
            return wrist.copy()
        r_aux = Rotation.from_quat(aux[3:])
        pos = aux[:3] + r_aux.apply(st["t_rel"])
        quat = (r_aux * Rotation.from_quat(st["r_rel"])).as_quat()
        return np.concatenate([pos, quat / np.linalg.norm(quat)])

    def update(self, poses, live, now=None):
        """poses: {physical serial: pose7}; live: {physical serial: bool}.

        Returns {out serial: pose7}; sides with both trackers lost hold their
        last output bit-exactly so the downstream freeze guard can trip.
        Source switches decay the static offset between the old output and the
        new source over ``switch_seconds`` — motion itself passes through at
        full rate, only the discontinuity is smoothed."""
        now = time.monotonic() if now is None else now
        out = {}
        for side, st in self.state.items():
            chosen = self._choose(st, poses, live)
            self.sources[side] = st["source"]
            if chosen is None:
                if st["wrist"] is not None:
                    out[self.out_serials[f"{side}_wrist"]] = st["wrist"]
                    out[self.out_serials[f"{side}_arm"]] = st["elbow"]
                continue
            if st["switch_from"] is not None:
                if st["switch_start"] is None:
                    st["switch_start"] = now
                    st["offset_pos"] = st["switch_from"][:3] - chosen[:3]
                    offset_rot = (Rotation.from_quat(st["switch_from"][3:])
                                  * Rotation.from_quat(chosen[3:]).inv())
                    st["offset_rot"] = offset_rot.as_rotvec()
                progress = min(1., (now - st["switch_start"]) / self.switch_seconds)
                decay = 1. - _smoothstep(progress)
                chosen = chosen.copy()
                chosen[:3] += decay * st["offset_pos"]
                rot = (Rotation.from_rotvec(decay * st["offset_rot"])
                       * Rotation.from_quat(chosen[3:]))
                chosen[3:] = rot.as_quat()
                if progress >= 1.:
                    st["switch_from"] = None
            if st["wrist"] is not None:
                chosen[3:] = _sign_fix(chosen[3:], st["wrist"][3:])
            elbow = chosen.copy()
            aux_pose = poses.get(st["sn_aux"])
            if live.get(st["sn_aux"], False) and aux_pose is not None and \
                    np.linalg.norm(aux_pose[:3] - chosen[:3]) > .05:
                axis = aux_pose[:3] - chosen[:3]
                axis_world = axis / np.linalg.norm(axis)
            elif st["axis_local"] is not None:
                axis = Rotation.from_quat(chosen[3:]).apply(st["axis_local"])
                axis_world = axis / np.linalg.norm(axis)
            elif st["wrist"] is not None:
                # No axis estimate yet: keep emitting the previous pair.
                out[self.out_serials[f"{side}_wrist"]] = st["wrist"]
                out[self.out_serials[f"{side}_arm"]] = st["elbow"]
                continue
            else:
                continue  # startup with dead aux: no elbow estimate possible yet
            elbow[:3] = chosen[:3] + axis_world * st["length"]
            st["wrist"], st["elbow"] = chosen, elbow
            out[self.out_serials[f"{side}_wrist"]] = chosen
            out[self.out_serials[f"{side}_arm"]] = elbow
        return out


def tracker_timestamp(sdk):
    """Prefer an available Motion timestamp, including SDKs exposing a zero stub."""
    motion_getter = getattr(sdk, "get_motion_timestamp_ns", None)
    motion_stamp = int(motion_getter()) if motion_getter is not None else 0
    if motion_stamp > 0:
        return "motion", motion_stamp
    return "overall", int(sdk.get_time_stamp_ns())


def read_tracker_frame(sdk):
    """Bracket separate SDK getters with the best available timestamp.

    Some app versions stream valid Motion poses without Motion.timeStampNs:
    the SDK getter exists but always returns zero. Use the overall timestamp
    in that case. That fallback measures packet freshness, not per-tracker
    freshness. A nonzero but stopped Motion timestamp must NOT fall back.
    """
    for _ in range(3):
        before = tracker_timestamp(sdk)
        serials = list(sdk.get_motion_tracker_serial_numbers())
        poses = list(sdk.get_motion_tracker_pose())
        count = int(sdk.num_motion_data_available())
        after = tracker_timestamp(sdk)
        if before != after:
            continue
        result = {}
        for i in range(min(max(count, 0), len(serials), len(poses))):
            pose = np.array(poses[i], dtype=float, copy=True)
            if pose.shape != (7,) or not np.isfinite(pose).all():
                continue
            norm = np.linalg.norm(pose[3:])
            if not 0.95 <= norm <= 1.05:
                continue
            pose[3:] /= norm
            result[str(serials[i])] = pose
        return result, after[1]
    return {}, 0


def __getattr__(name):
    """Lazily expose model-frame constants so this module imports without the
    vendor CAD.  ``ROBOT_TO_SIM`` is the operator X-forward/Y-left/Z-up frame
    mapped into the assembly URDF world."""
    if name in ("ROBOT_TO_SIM", "ROBOT_TO_MODEL"):
        value = marvin_model.ROBOT_TO_MODEL
        globals()["ROBOT_TO_SIM"] = value
        return value
    if name == "DEFAULT_ALIGNMENT":
        value = default_alignment_path()
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
