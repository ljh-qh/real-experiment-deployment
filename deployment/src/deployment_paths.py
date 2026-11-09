"""All deployment resources resolve inside this checkout."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def robot_env():
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONPATH'] = ':'.join(str(ROOT / p) for p in (
        'src', 'vendor/bridge', 'deps/infra', 'deps/observer',
        'vendor/universal_robot_infra/src', 'assets/robot',
        'vendor/mujoco_marvin', 'sdk/sharpa/python'))
    env['MARVIN_SDK_DIR'] = str(ROOT / 'sdk/tianji')
    env['SHARPA_SDK_PY'] = str(ROOT / 'sdk/sharpa/python')
    return env
