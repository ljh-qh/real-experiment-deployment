"""Offline home-pose comparison using the production model and IK.

No SDK is initialized. The meshes' collisions and dynamics are not validated;
these results describe kinematic tracking and joint margin only. Hand mounts
stay fixed throughout. Run with the simulator environment, optionally with
--search:

    python -m mujoco_marvin.home_pose_study --output report.json [--search]
"""

import argparse
import contextlib
import hashlib
import io
import itertools
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .robot_description import load_home_joints, resolve_model_urdf
from .teleop import MarvinTeleop, build_model, dispatch_command
from .tracking import ROLES


MIRROR_JOINTS = np.array([-1, 1, -1, 1, -1, 1, -1])


def poses():
    return {r: np.array([0., 0., 0., 0., 0., 0., 1.]) for r in ROLES}


def make_controller(model, home, gain=.35):
    from .tracking import ROBOT_TO_SIM

    model.key("home").qpos[:] = np.deg2rad(home['left'] + home['right'])
    model.key("home").ctrl[:] = model.key("home").qpos
    unit = MarvinTeleop(model, {r: r for r in ROLES}, ROBOT_TO_SIM, elbow_gain=gain)
    with contextlib.redirect_stdout(io.StringIO()):
        unit.accept(poses(), 1)
    return unit


def frame_targets(position, rotation, elbow):
    sample = poses()
    for side, sign in (("left", 1), ("right", -1)):
        mirror = np.array([1, sign, 1])
        sample[f"{side}_wrist"][:3] = position * mirror
        sample[f"{side}_wrist"][3:] = Rotation.from_rotvec(rotation * [sign, 1, sign]).as_quat()
        sample[f"{side}_arm"][:3] = elbow * mirror
    return sample


def errors_and_margin(unit):
    rows = []
    for side in ("left", "right"):
        site = unit.site_ids[side]["tcp"]
        target_pos, target_rot, _ = unit.targets[side]
        actual_rot = Rotation.from_matrix(unit.data.site_xmat[site].reshape(3, 3))
        pe = np.linalg.norm(unit.data.site_xpos[site] - target_pos) * 1000
        re = np.degrees((target_rot * actual_rot.inv()).magnitude())
        q = unit.command[unit.qpos_ids[side]]
        limits = unit.model.jnt_range[unit.joint_ids[side]]
        margin = np.degrees(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))
        rows.append([float(pe), float(re), float(margin)])
    return np.array(rows)


def home_metrics(unit):
    from .tracking import ROBOT_TO_SIM

    result = {}
    model, data = unit.model, unit.data
    for side in ("left", "right"):
        jp = np.zeros((3, model.nv)); jr = np.zeros_like(jp)
        mujoco.mj_jacSite(model, data, jp, jr, unit.site_ids[side]["tcp"])
        dofs = unit.dof_ids[side]
        singular = np.linalg.svd(np.vstack((jp[:, dofs], .5 * jr[:, dofs])), compute_uv=False)
        shoulder = data.xpos[model.body(f"Link1_{side[0].upper()}").id]
        pose = data.site_xpos[unit.site_ids[side]["tcp"]]
        elbow = data.site_xpos[unit.site_ids[side]["elbow"]]
        result[side] = {
            'condition_number': float(singular[0] / singular[-1]),
            'min_singular_value': float(singular[-1]),
            'wrist_from_shoulder_m': (ROBOT_TO_SIM.T @ (pose - shoulder)).tolist(),
            'elbow_from_shoulder_m': (ROBOT_TO_SIM.T @ (elbow - shoulder)).tolist(),
        }
    return result


def discrete_cases(model, home, gain):
    cases = []
    for axis in range(3):
        for sign in (-1, 1):
            cases.append((f"position_{axis}_{sign}", np.eye(3)[axis] * .15 * sign, np.zeros(3)))
            cases.append((f"rotation_{axis}_{sign}", np.zeros(3), np.eye(3)[axis] * np.deg2rad(25) * sign))
    for dx, dy, dz in itertools.product([-.1, .1], repeat=3):
        cases.append((f"combined_{dx}_{dy}_{dz}", np.array([dx, dy, dz]),
                      np.deg2rad([12 * np.sign(dy), -15 * np.sign(dx), 15 * np.sign(dz)])))
    results = []
    for label, dp, dr in cases:
        unit = make_controller(model, home, gain)
        minimum_margin = 180.
        for tick in range(60):
            alpha = min(tick / 35, 1.)
            sample = frame_targets(alpha * dp, alpha * dr, alpha * np.array([-.12, .12, -.08]))
            unit.accept(sample, tick + 2); dispatch_command(unit)
            minimum_margin = min(minimum_margin, float(errors_and_margin(unit)[:, 2].min()))
        errors = errors_and_margin(unit)
        results.append({'case': label, 'errors_mm_deg': errors[:, :2].tolist(),
                        'min_margin_deg': minimum_margin})
    errors = np.array([r['errors_mm_deg'] for r in results]).reshape(-1, 2)
    return {'passed': int(np.sum((errors[:, 0] < 5) & (errors[:, 1] < 2))),
            'evaluations': len(errors), 'max_error_mm': float(errors[:, 0].max()),
            'max_error_deg': float(errors[:, 1].max()),
            'min_margin_deg': min(r['min_margin_deg'] for r in results), 'cases': results}


def continuous_case(model, home, gain, samples=1441):
    unit = make_controller(model, home, gain)
    errors = []
    margins = {s: np.full(7, np.inf) for s in ('left', 'right')}
    limit_hits = {s: np.zeros(7, dtype=int) for s in ('left', 'right')}
    for tick in range(samples):
        t = 2 * np.pi * tick / 240.
        ramp = min(tick / 90, 1.)
        position = ramp * np.array([.12 * np.sin(t), .14 * np.sin(.5 * t), .12 * np.sin(.75 * t)])
        rotation = ramp * np.deg2rad([15 * np.sin(.5 * t), 20 * np.sin(.75 * t), 20 * np.sin(t)])
        elbow = ramp * np.array([-.08 * (1 - np.cos(.5 * t)), .10 * np.sin(.5 * t), -.05 * (1 - np.cos(.75 * t))])
        unit.accept(frame_targets(position, rotation, elbow), tick + 2)
        dispatch_command(unit)
        errors.extend(errors_and_margin(unit).tolist())
        for side in ('left', 'right'):
            q = unit.command[unit.qpos_ids[side]]
            limits = model.jnt_range[unit.joint_ids[side]]
            margin = np.degrees(np.minimum(q - limits[:, 0], limits[:, 1] - q))
            margins[side] = np.minimum(margins[side], margin)
            limit_hits[side] += margin < .1
    values = np.array(errors)
    return {'samples_per_arm': samples, 'nominal_hz': 90, 'elbow_gain': gain,
            'p95_error_mm': float(np.percentile(values[:, 0], 95)),
            'max_error_mm': float(values[:, 0].max()),
            'p95_error_deg': float(np.percentile(values[:, 1], 95)),
            'max_error_deg': float(values[:, 1].max()),
            'min_margin_deg': float(values[:, 2].min()),
            'arm_samples_within_0_1_deg_of_limit': int(np.sum(values[:, 2] < .1)),
            'per_joint_min_margin_deg': {s: a.tolist() for s, a in margins.items()},
            'per_joint_limit_hit_samples': {s: a.tolist() for s, a in limit_hits.items()}}


def translation_rays(model, home, gain):
    results = []
    for axis, sign in itertools.product(range(3), (-1, 1)):
        unit = make_controller(model, home, gain)
        reach = 0.; failure = None
        for stage, distance in enumerate(np.arange(.025, .301, .025)):
            for tick in range(36):
                stroke = distance - .025 + .025 * min(tick / 20, 1.)
                sample = frame_targets(np.eye(3)[axis] * sign * stroke, np.zeros(3), np.array([-.12, .12, -.08]))
                unit.accept(sample, 2 + stage * 36 + tick); dispatch_command(unit)
            errors = errors_and_margin(unit)
            if np.any(errors[:, 0] > 5) or np.any(errors[:, 1] > 2):
                failure = errors[:, :2].tolist()
                break
            reach = distance
        results.append({'axis': 'XYZ'[axis], 'sign': sign, 'reach_cm': round(100 * reach, 1),
                        'capped_at_30_cm': bool(reach >= .30 - 1e-9), 'first_failure_mm_deg': failure})
    return results


def search_candidates(model, baseline):
    from .tracking import ROBOT_TO_SIM

    unit = make_controller(model, baseline)
    data = unit.data
    side = 'left'; qids = unit.qpos_ids[side]; tcp = unit.site_ids[side]['tcp']
    shoulder = data.xpos[model.body('Link1_L').id].copy()
    rotation = Rotation.from_matrix(data.site_xmat[tcp].reshape(3, 3))
    seed = data.qpos[qids].copy(); limits = model.jnt_range[unit.joint_ids[side]]
    candidates = []
    for x, y, z, q2 in itertools.product([.36, .40, .44, .48], [.04, .08], [-.26, -.31, -.36], [55, 70, 85]):
        position = shoulder + ROBOT_TO_SIM @ [x, y, z]
        preferred = seed.copy(); preferred[1] = np.deg2rad(q2)
        def residual(q):
            data.qpos[qids] = q; mujoco.mj_forward(model, data)
            error_rot = (Rotation.from_matrix(data.site_xmat[tcp].reshape(3, 3)) * rotation.inv()).as_rotvec()
            return np.r_[6 * (data.site_xpos[tcp] - position), error_rot,
                         .18 * (q[1] - preferred[1]), .002 * (q - preferred)]
        fitted = least_squares(residual, preferred, bounds=(limits[:, 0] + .12, limits[:, 1] - .12),
                               max_nfev=80, ftol=1e-7, xtol=1e-7, gtol=1e-7)
        data.qpos[qids] = fitted.x; mujoco.mj_forward(model, data)
        pose_error = np.linalg.norm(data.site_xpos[tcp] - position)
        angle_error = np.degrees((Rotation.from_matrix(data.site_xmat[tcp].reshape(3, 3)) * rotation.inv()).magnitude())
        elbow = ROBOT_TO_SIM.T @ (data.site_xpos[unit.site_ids[side]['elbow']] - shoulder)
        if pose_error > .001 or angle_error > .1 or not (-.02 <= elbow[1] <= .22 and -.08 <= elbow[0] <= .26):
            continue
        jp = np.zeros((3, model.nv)); jr = np.zeros_like(jp)
        mujoco.mj_jacSite(model, data, jp, jr, tcp)
        sv = np.linalg.svd(np.vstack((jp[:, qids], .5 * jr[:, qids])), compute_uv=False)
        margin = np.degrees(np.min(np.minimum(fitted.x - limits[:, 0], limits[:, 1] - fitted.x)))
        candidates.append({'wrist_from_shoulder_m': [x, y, z], 'q2_preference_deg': q2,
                           'left_deg': np.degrees(fitted.x).round(4).tolist(),
                           'min_singular_value': float(sv[-1]), 'home_margin_deg': float(margin),
                           'elbow_from_shoulder_m': elbow.tolist(),
                           'screening_score': float(sv[-1] * min(margin / 45, 1.))})
    return sorted(candidates, key=lambda r: r['screening_score'], reverse=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('.local/home-pose-study/report.json'))
    parser.add_argument('--search', action='store_true', help='Also report the 72-point candidate grid')
    args = parser.parse_args(argv)
    model = build_model()
    baseline = load_home_joints(profile='elbows_90')
    urdf = resolve_model_urdf()
    report = {'urdf': str(urdf), 'urdf_sha256': hashlib.sha256(urdf.read_bytes()).hexdigest(),
              'orientation_weight_m_per_rad': .5, 'tested_elbow_gains': [0., .35],
              'limits': 'URDF joint limits; no collision/dynamics or hardware validation',
              'fixed_hand_mounts': True, 'profiles': {}}
    if args.search:
        report['search_candidates'] = search_candidates(model, baseline)
        print(f"Accepted {len(report['search_candidates'])}/72 grid candidates", flush=True)
    for name, home in [('elbows_90', baseline), ('default', load_home_joints())]:
        profile = {'joints_deg': home, 'home': home_metrics(make_controller(model, home)), 'by_gain': {}}
        for gain in (0., .35):
            outcomes = {'discrete': discrete_cases(model, home, gain),
                        'continuous': continuous_case(model, home, gain),
                        'rays': translation_rays(model, home, gain)}
            profile['by_gain'][str(gain)] = outcomes
            print(name, gain, outcomes['continuous'], flush=True)
        report['profiles'][name] = profile
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(args.output, flush=True)


if __name__ == '__main__':
    main()
