"""RealSense + measured joints -> ZMQ; explicit opt-in left model probe."""
import argparse
from collections import Counter
import json
from pathlib import Path
import threading
import time
import uuid

import numpy as np
import zmq

from snapshot_publisher import SnapshotPublisher, snapshot_from_measured
from shadow_observation import prepare_snapshot
from zmq_chunk_worker import loopback_endpoint
from zmq_offline_bridge import load_codec


class CameraFeed:
    def __init__(self, serial, rotate):
        import pyrealsense2 as rs
        self.rs = rs
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        self.pipeline.start(cfg)
        self.rotate = rotate
        self.latest = None
        self.error = None
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        try:
            while not self.stop.is_set():
                frames = self.pipeline.wait_for_frames(1000)
                frame = frames.get_color_frame()
                if not frame:
                    continue
                # Stamp host receipt before conversion/rotation, so processing
                # delay cannot make an older camera frame look freshly captured.
                stamp = time.monotonic_ns()
                image = np.asanyarray(frame.get_data()).copy()
                if self.rotate:
                    image = np.rot90(image, self.rotate // 90).copy()
                with self.lock:
                    self.latest = (image, stamp)
        except Exception as exc:
            self.error = str(exc)

    def snapshot(self):
        if self.error is not None:
            raise RuntimeError(self.error)
        with self.lock:
            return self.latest

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
        self.pipeline.stop()


def read_hands(hands):
    from mujoco_marvin.sharpa_backend import read_hand_rad
    values, stamps = [], {}
    for side in ('left', 'right'):
        values.append(read_hand_rad(hands[side]))
        stamps[side + '_hand'] = time.monotonic_ns()
    return np.concatenate(values), stamps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot-ip', required=True)
    parser.add_argument('--sdk-dir', type=Path, required=True)
    parser.add_argument('--left-hand-sn', required=True)
    parser.add_argument('--right-hand-sn', required=True)
    parser.add_argument('--main-sn', required=True)
    parser.add_argument('--left-wrist-sn', required=True)
    parser.add_argument('--right-wrist-sn', required=True)
    parser.add_argument('--endpoint', type=loopback_endpoint, required=True)
    parser.add_argument('--control-endpoint', type=loopback_endpoint, required=True)
    parser.add_argument('--codec', type=Path, required=True)
    parser.add_argument('--urdf', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--duration', type=float, default=180)
    parser.add_argument('--allow-left-model-probe-50mm', action='store_true')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    from mujoco_marvin.tianji_backend import TianjiJointBackend
    from mujoco_marvin.sharpa_backend import load_sharpa, discover
    from kinematics import URDFKinematics
    robot = None
    manager = None
    connected_sn = []
    cameras = {}
    publisher = None
    context = zmq.Context()
    control = context.socket(zmq.REP)
    control.setsockopt(zmq.LINGER, 0)
    control.bind(args.control_endpoint)
    codec = load_codec(args.codec)
    kinematics = URDFKinematics(args.urdf)
    stats = Counter()
    stream_id = str(uuid.uuid4())
    last_snapshot = None
    hand_modes = {}
    try:
        # initialize=False is essential: normal initialization ramps both arms home.
        robot = TianjiJointBackend(args.robot_ip, sdk_dir=str(args.sdk_dir), initialize=False)
        if not robot._verify_connection():
            raise RuntimeError('Arm telemetry is not advancing')
        sdk = load_sharpa()
        manager, devices = discover(sdk, timeout=5)
        by_sn = {device.sn: device for device in devices}
        hands = {}
        for side, serial in (('left', args.left_hand_sn), ('right', args.right_hand_sn)):
            if serial not in by_sn or by_sn[serial].hand_side != getattr(sdk.HandSide, side.upper()):
                raise RuntimeError(f'{side} hand serial/side mismatch')
            hands[side] = manager.connect(serial, skip_tactile=True)
            connected_sn.append(serial)
            hand_modes[side] = {}
            for name in ('control_source', 'control_mode', 'enable_state'):
                error, value = getattr(hands[side], 'get_' + name)()
                if int(error.code) != 0:
                    raise RuntimeError(f'{side} {name} read failed')
                hand_modes[side][name] = str(value)
        for role, serial, rotate in (('main', args.main_sn, 0),
                ('wrist2', args.left_wrist_sn, 90), ('wrist', args.right_wrist_sn, 90)):
            cameras[role] = CameraFeed(serial, rotate)
        publisher = SnapshotPublisher(args.endpoint, codec)
        start, next_tick = time.monotonic(), time.monotonic()
        ready = False
        with (args.output_dir / 'measured_joints.jsonl').open('w') as measurements:
            while time.monotonic() - start < args.duration:
                if control.poll(0, zmq.POLLIN):
                    request = control.recv_json()
                    if request['type'] == 'shutdown':
                        control.send_json({'type': 'shutdown_ack', 'hardware_commands_sent': 0})
                        break
                    elif request['type'] == 'snapshot':
                        control.send_json({'type': 'snapshot', 'data': last_snapshot,
                                           'hardware_commands_sent': 0})
                    elif request['type'] == 'left_model_probe' and args.allow_left_model_probe_50mm:
                        age_ms=(time.monotonic_ns()-int(request['observation_timestamp_ns']))/1e6
                        if age_ms < 0 or age_ms > 1000:
                            control.send_json({'type':'error','message':'Model observation is stale'})
                            continue
                        from bounded_model_motion import execute_model_probe
                        result=execute_model_probe(robot,kinematics,request['actions'])
                        (args.output_dir/'model_motion.json').write_text(json.dumps(result,indent=2))
                        control.send_json({'type':'motion_result','result':{
                            k:v for k,v in result.items() if k!='samples'}})
                        if result['status'] != 'completed':
                            raise RuntimeError('Model probe aborted; see model_motion.json')
                    else:
                        control.send_json({'type': 'error', 'message': 'Read-only control endpoint'})
                captured = {role: feed.snapshot() for role, feed in cameras.items()}
                if any(value is None for value in captured.values()):
                    time.sleep(.01)
                    continue
                left, right = robot.read_joints_deg()
                arm_stamp = time.monotonic_ns()
                fingers, stamps = read_hands(hands)
                stamps.update(left_arm=arm_stamp, right_arm=arm_stamp)
                stamps.update({role: item[1] for role, item in captured.items()})
                raw = snapshot_from_measured(stream_id, stats['valid'], np.deg2rad(np.r_[left, right]),
                    fingers, {role: item[0] for role, item in captured.items()}, stamps)
                try:
                    state = prepare_snapshot(raw, kinematics, 'live')
                except ValueError as exc:
                    stats['rejected'] += 1
                    stats[str(exc)] += 1
                else:
                    publisher.offer(raw)
                    last_snapshot = {'timestamp_ns': arm_stamp, 'left_deg': left.tolist(),
                        'right_deg': right.tolist(), 'hand_rad': fingers.tolist(),
                        'state80': state['observation']['state'].tolist(),
                        'skew_ms': state['modality_skew_ms']}
                    measurements.write(json.dumps(last_snapshot) + '\n')
                    stats['valid'] += 1
                    if not ready:
                        metadata = {'pid': __import__('os').getpid(), 'source_kind': 'live',
                            'endpoint': publisher.endpoint,
                            'control_endpoint': control.getsockopt_string(zmq.LAST_ENDPOINT),
                            'hand_modes_unchanged': hand_modes, 'hardware_commands_sent': 0}
                        (args.output_dir / 'ready.json').write_text(json.dumps(metadata, indent=2))
                        for role, item in captured.items():
                            from PIL import Image
                            Image.fromarray(item[0]).save(args.output_dir / (role + '.png'))
                        print(json.dumps({'type': 'ready', **metadata}), flush=True)
                        ready = True
                next_tick += 1 / 30
                time.sleep(max(0, next_tick - time.monotonic()))
                if next_tick < time.monotonic() - .1:
                    next_tick = time.monotonic()
        if not ready:
            raise RuntimeError(f'No complete fresh observation: {dict(stats)}')
    finally:
        if publisher is not None:
            publisher.close()
        for feed in cameras.values():
            feed.close()
        if manager is not None:
            for serial in connected_sn:
                manager.disconnect(serial)
        if robot is not None:
            robot.close()  # _motion_started=False: release only, no set_state.
        control.close()
        context.term()
        (args.output_dir / 'summary.json').write_text(json.dumps({
            'stats': dict(stats), 'source_kind': 'live', 'hardware_commands_sent': 0,
            'hand_initial_modes': hand_modes}, indent=2))


if __name__ == '__main__':
    main()
