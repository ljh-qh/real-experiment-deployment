"""Unified launcher. Checking profiles never initializes hardware or CUDA."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import yaml
from deployment_paths import ROOT, robot_env


def read_profile(path, output):
    def resolve(value):
        if isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(v) for v in value]
        if isinstance(value, str):
            return value.replace('$DEPLOY', str(ROOT)).replace('$OUTPUT', str(output))
        return value
    return resolve(yaml.safe_load(Path(path).read_text()))


def origin_reference(cfg):
    if "origin_reference" in cfg:
        return Path(cfg["origin_reference"])
    return ROOT / "assets/references/badminton_demo_origin.json"


def check_profile(cfg):
    worker = cfg['worker']
    if 'task' in cfg:
        if cfg['task'] not in ('badminton', 'top', 'bottle') or (worker['args']['task'] if cfg['model'] == 'openwam' else ('bottle' if worker['args']['version'] == 'bottle' else 'badminton')) != cfg['task']:
            raise ValueError('Client and worker task mismatch')
        if 'origin_reference' not in cfg:
            raise ValueError('Explicit task requires its origin_reference')
        if cfg['task'] in ('top', 'bottle'):
            if cfg['model'] not in ('openwam','easywam') or cfg['client_entrypoint'] != ('confirmed_client.py' if 'rtc' in cfg else 'accepted_client.py'):
                raise ValueError('Task profile requires the OpenWAM state client')
            if cfg['model'] == 'openwam' and Path(worker['args']['checkpoint']).name != 'openwam_' + cfg['task']:
                raise ValueError('Task and checkpoint directory mismatch')
            reference = json.loads(origin_reference(cfg).read_text())
            if reference['task'] != cfg['task']:
                raise ValueError('Task and demonstration origin mismatch')
    paths = [worker[k] for k in ('python', 'script', 'cwd')]
    paths += worker['env']['PYTHONPATH'].split(':')
    paths += [v for k, v in worker['args'].items() if k in ('config', 'checkpoint', 'codec', 'warmup-fixtures')]
    paths += [str(ROOT / 'src' / cfg['client_entrypoint']),
              str(origin_reference(cfg))]
    if 'checkpoint' in worker['args']:
        paths.append(str(Path(worker['args']['checkpoint']) / worker['args']['checkpoint-name']))
    if cfg['model'] == 'easywam' and worker['args']['version'] == 'bottle':
        paths += [str(ROOT / 'assets/checkpoints/easywam_bottle' / n) for n in ('step_008205.pt','deployment_model.yaml','normalization_stats_62.npy','bottle_easywam_context.pt')]
        paths += [str(ROOT / 'assets/base_models/Cosmos-Predict2.5-2B')]
    elif cfg['model'] == 'easywam':
        paths += [str(ROOT / 'assets' / p) for p in (
            'checkpoints/easywam_badminton/step_004465.pt',
            'checkpoints/easywam_badminton/config.yaml',
            'checkpoints/easywam_common/normalization_stats_62.npy',
            'checkpoints/easywam_common/badminton_context.pt',
            'base_models/Cosmos-Predict2.5-2B')]
    for value in paths:
        path = Path(value).resolve()
        # System Python is the documented venv interpreter dependency.
        if value == worker['python']:
            path = Path(value).absolute()
        if not path.is_relative_to(ROOT):
            raise ValueError(f'Runtime path escapes deployment: {value}')
        if not Path(value).exists():
            raise FileNotFoundError(value)
    if not cfg['client']['sync']:
        if 'async_prefetch' in cfg:
            if cfg['client_entrypoint'] != 'accepted_client.py' or 'rtc' in cfg:
                raise ValueError('Plain prefetch requires the state client without RTC')
            gate = json.loads(Path(cfg['async_prefetch']['validation_file']).read_text())
            if gate['status'] != 'passed' or gate['task'] != cfg['task']:
                raise RuntimeError('Async model validation failed')
            if not 0 < cfg['client']['prefetch_delay'] < 32:
                raise ValueError('Prefetch must leave an unexecuted action suffix')
        else:
            gate = json.loads(Path(cfg['rtc']['validation_file']).read_text())
            if not gate['hardware_ready']:
                raise RuntimeError('RTC gate: ' + gate['reason'])
    return {'profile_valid': True, 'model': cfg['model'],
            'paths_checked': len(paths), 'hardware_connected': False, 'gpu_used': False}


def stop_worker(proc):
    if proc is not None:
        # Popen created this session; its compiler children belong to the same group.
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        finally:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def start_worker(cfg):
    worker = cfg['worker']
    args = []
    for key, value in worker['args'].items():
        if value is True:
            args.append('--' + key)
        elif value is not False and value is not None:
            args.extend(['--' + key, str(value)])
    env = os.environ.copy()
    env.update({key: str(value) for key, value in worker['env'].items()})
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    ready = Path(worker['args']['ready-file'])
    ready.unlink(missing_ok=True)
    proc = None
    try:
        with ready.with_suffix('.worker.log').open('w') as log:
            proc = subprocess.Popen([worker['python'], worker['script'], *args],
                                    cwd=worker['cwd'], env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        (ready.parent / 'worker.pid').write_text(str(proc.pid))
        deadline = time.monotonic() + 360
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f'Worker exited ({proc.returncode}); inspect {ready.with_suffix(".worker.log")}')
            if ready.exists():
                return proc, ready
            time.sleep(1)
        raise TimeoutError('Worker did not become ready in 360 seconds')
    except BaseException:
        stop_worker(proc)
        raise


def confirm_stop(output):
    from mujoco_marvin.tianji_backend import TianjiJointBackend
    from return_full_origin import confirm_idle
    backend = None
    result = {'motion_commands_sent': 0, 'scope': 'arms'}
    try:
        backend = TianjiJointBackend('192.168.1.190', sdk_dir=str(ROOT / 'sdk/tianji'), initialize=False)
        if not backend._verify_connection():
            raise RuntimeError('Feedback not advancing')
        result['before'] = [int(x['cur_state']) for x in backend._subscribe()['states']]
        result['idle_requests'] = confirm_idle(backend, timeout=6.)
        result['after'] = [int(x['cur_state']) for x in backend._subscribe()['states']]
        result['status'] = 'idle_confirmed'
    except Exception as error:
        result.update(status='error', error=str(error))
    finally:
        if backend:
            try:
                backend.close()
            except Exception as error:
                result.update(status='error', close_error=str(error))
        (output / 'stop_confirmation.json').write_text(json.dumps(result, indent=2))
    return result


def merge_motion_shutdown(output, result, report_name):
    """Keep arm idle acknowledgement separate from the motion child's cleanup."""
    result = dict(result)
    result['arm_status'] = result['status']
    path = output / report_name
    result['motion_report'] = str(path)
    if not path.is_file():
        result.update(status='error', hands_status='unconfirmed',
                      shutdown_error='Motion child did not produce a shutdown report')
    else:
        try:
            motion = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            result.update(status='error', hands_status='unconfirmed',
                          shutdown_error=f'Motion shutdown report unreadable: {error}')
            (output / 'stop_confirmation.json').write_text(json.dumps(result, indent=2))
            return result
        if not isinstance(motion, dict):
            result.update(status='error', hands_status='unconfirmed',
                          shutdown_error='Motion shutdown report must be an object')
            (output / 'stop_confirmation.json').write_text(json.dumps(result, indent=2))
            return result
        errors = [str(motion[key]) for key in ('shutdown_error', 'hand_shutdown_error',
                                             'backend_close_error', 'manager_disconnect_errors') if motion.get(key)]
        if report_name == 'result.json':
            hands = motion.get('hand_shutdown', {})
            closed = (isinstance(hands, dict) and set(hands) == {'left', 'right'}
                      and all(isinstance(v, dict) and v.get('status') == 'closed' for v in hands.values()))
            result['hand_shutdown'] = hands
        else:
            closed = motion.get('hands_shutdown') == 'disabled; SDK source'
        result['hands_status'] = 'closed' if closed else 'unconfirmed'
        if errors or not closed:
            result['status'] = 'error'
            result['shutdown_error'] = '; '.join(errors) if errors else 'Both hands were not confirmed closed'
    (output / 'stop_confirmation.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description='Local deployment; --check is the default. --run moves the robot.')
    parser.add_argument('--profile', default='openwam_sync', help='Profile name or YAML path')
    parser.add_argument('--output', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='Offline path/config check only (default)')
    mode.add_argument('--run', action='store_true', help='Load model, return robot to origin, then execute')
    mode.add_argument('--worker-only', action='store_true', help='Run inference worker until Ctrl+C; no robot')
    args = parser.parse_args()
    profile = Path(args.profile)
    if profile.suffix != '.yaml':
        profile = ROOT / 'configs/profiles' / (args.profile + '.yaml')
    output = (args.output or ROOT / 'runs' / (profile.stem + '_' + time.strftime('%Y%m%d_%H%M%S'))).resolve()
    if not output.is_relative_to(ROOT):
        parser.error('--output must be inside deployment/')
    cfg = read_profile(profile, output)
    print(json.dumps(check_profile(cfg), ensure_ascii=False), flush=True)
    if not args.run and not args.worker_only:
        return 0
    # Do not compete silently with another GPU task.
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'],
                         check=True, text=True, capture_output=True)
    if gpu.stdout.strip():
        raise RuntimeError('GPU already has compute processes; resolve resource availability first:\n' + gpu.stdout)
    lock = (ROOT / 'runs/.active.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'model_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    (output / 'runner.pid').write_text(str(os.getpid()))
    (output / 'runner_identity.json').write_text(json.dumps({
        'pid': os.getpid(), 'start_ticks': Path('/proc/self/stat').read_text().split()[21]}))
    env = robot_env()
    # This process only imports hardware for final idle confirmation after a motion child was started.
    os.environ.update({k: env[k] for k in ('MARVIN_SDK_DIR', 'SHARPA_SDK_PY')})
    sys.path[:0] = env['PYTHONPATH'].split(':')
    python = str(ROOT / 'envs/robot/bin/python')
    worker = child = None
    motion_started = False
    motion_report_name = 'return.json'
    rc = 0

    def stage(name, **fields):
        value = {'stage': name, **fields}
        (output / 'stage.json').write_text(json.dumps(value, indent=2))
        print(json.dumps(value), flush=True)

    def interrupt(*_):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    try:
        stage('loading_worker')
        worker, ready = start_worker(cfg)
        if args.worker_only:
            stage('worker_ready', worker_pid=worker.pid, hardware_connected=False)
            rc = worker.wait()
        else:
            stage('returning_to_origin', worker_pid=worker.pid)
            with (output / 'return.log').open('w') as log:
                child = subprocess.Popen([python, str(ROOT / 'vendor/bridge/return_full_origin.py'),
                    '--root', str(ROOT), '--reference', str(origin_reference(cfg)),
                    '--output', str(output / 'return.json'), '--execute'],
                    env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                motion_started = True
                (output / 'motion_child.pid').write_text(str(child.pid))
                if child.wait():
                    raise RuntimeError('Return to origin failed; inspect return.json / return.log')
            client = cfg['client']
            command = [python, str(ROOT / 'src' / cfg['client_entrypoint']),
                       '--worker-ready', str(ready), '--output', str(output)]
            for key in ('duration', 'hz', 'prefetch_delay', 'max_joint_step_deg', 'max_hand_step_deg',
                        'feedback_confirm_ms', 'arm_error_deg', 'hand_error_deg', 'rtc_commit_steps', 'rtc_prefix_capacity'):
                if key in client:
                    command.extend(['--' + key.replace('_', '-'), str(client[key])])
            if 'task' in cfg:
                command.extend(['--task', cfg['task'], '--origin-reference', str(origin_reference(cfg))])
            if client['sync']:
                command.append('--sync')
            motion_report_name = 'result.json'
            with (output / 'client.log').open('w') as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                (output / 'motion_child.pid').write_text(str(child.pid))
                (output / 'client.pid').write_text(str(child.pid))
                stage('client_running', worker_pid=worker.pid, client_pid=child.pid, unlimited=client['duration'] <= 0)
                rc = child.wait()
            result = json.loads((output / 'result.json').read_text())
            if result['status'] != 'completed' or result.get('shutdown_error'):
                rc = rc or 1
            stage('client_exited', client_rc=rc, client_status=result['status'], error=result.get('error'))
    except KeyboardInterrupt:
        rc = 130
        stage('stop_requested')
    finally:
        # Finish reclaiming the motion process before sending the idle confirmation.
        old_handlers = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            if child is not None and child.poll() is None:
                child.send_signal(signal.SIGINT)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
        finally:
            try:
                stop_worker(worker)
                result = confirm_stop(output) if motion_started else {'hardware_connected': False}
                if motion_started:
                    result = merge_motion_shutdown(output, result, motion_report_name)
                if motion_started and result.get('status') != 'idle_confirmed':
                    rc = rc or 1
                stage('stopped', stop_confirmation=result)
            finally:
                lock.close()
                for sig, handler in old_handlers.items():
                    signal.signal(sig, handler)
    return rc
