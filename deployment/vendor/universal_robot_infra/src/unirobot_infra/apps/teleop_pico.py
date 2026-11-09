"""PICO 遥操统一入口（原 ``mujoco-marvin`` CLI 的替代）。

主流程用四个 Object tracker，子命令与原子命令一一对应::

    unirobot-infra-teleop-pico sim       --backend sim (Object tracker 输入)
    unirobot-infra-teleop-pico calibrate --calibrate --backend sim (Object 模式)
    unirobot-infra-teleop-pico arms      --backend robot   (需要机械臂 IP)
    unirobot-infra-teleop-pico both      --backend both    (需要机械臂 IP)
    unirobot-infra-teleop-pico pico      --input-source body --backend sim (Full Body)
    unirobot-infra-teleop-pico body      Full Body 入口（懒加载 teleop_pico_body.run）
    unirobot-infra-teleop-pico preview   --input-source body --preview-body --backend sim

子命令之后的参数原样透传给遥操核心，例如::

    unirobot-infra-teleop-pico pico --demo --headless --duration 2

pico/both/arms/sim/preview/calibrate 解析为带前缀的 argv 后委托
``unirobot_infra.teleop.marvin_teleop.main``；body 子命令懒加载
``unirobot_infra.apps.teleop_pico_body``（Full Body 链，切片 5 迁入，
尚未落位前 import 会自然报错）并直接调用其 ``run(args)``。

机械臂 IP 解析优先级：``--robot-ip`` 参数 > ``TIANJI_ROBOT_IP`` 环境变量 >
``configs/robot/marvin_robot.yaml`` 的 ``robot_ip``；arms/both 缺少 IP 时拒绝启动。

也可 ``python -m unirobot_infra.apps.teleop_pico <子命令> ...`` 运行。
"""

import argparse
import os

from unirobot_infra.robots.arm.marvin_model import resolve_robot_config
from unirobot_infra.teleop import marvin_teleop

_MODES = (
    ("pico", "PICO teleop of the sim via the separate 5-point Full Body skeleton"),
    ("body", "Full Body 入口：懒加载 unirobot_infra.apps.teleop_pico_body.run"),
    ("preview", "Read-only Full Body skeleton preview"),
    ("sim", "Teleoperate the MuJoCo model only (--backend sim); Object tracker input"),
    ("calibrate", "Interactive tracker direction calibration, then sim teleop (Object mode)"),
    ("arms", "Stream IK to the real Tianji arms (--backend robot)"),
    ("both", "Sim viewer and real Tianji arms together (--backend both)"),
)


def _has_robot_ip(extra):
    return "--robot-ip" in extra or any(a.startswith("--robot-ip=") for a in extra)


def _config_robot_ip():
    try:
        import yaml
        cfg = yaml.safe_load(resolve_robot_config().read_text())
    except Exception:
        return None
    ip = (cfg or {}).get("robot_ip")
    return ip if isinstance(ip, str) and ip.strip() and not ip.strip().startswith("YOUR_") else None


def build_teleop_argv(mode, extra, parser):
    if mode == "sim":
        return ["--backend", "sim", *extra]
    if mode == "calibrate":
        return ["--backend", "sim", "--calibrate", *extra]
    if mode in ("arms", "both"):
        argv = []
        if not _has_robot_ip(extra):
            ip = os.environ.get("TIANJI_ROBOT_IP", "").strip() or _config_robot_ip()
            if not ip:
                parser.error(
                    f"'{mode}' output requires a robot IP: pass --robot-ip <ip>, "
                    "set TIANJI_ROBOT_IP, or configure robot_ip in marvin_robot.yaml")
            argv += ["--robot-ip", ip]
        argv += ["--backend", "robot" if mode == "arms" else "both", *extra]
        return argv
    if mode in ("pico", "body"):
        return ["--input-source", "body", "--backend", "sim", *extra]
    if mode == "preview":
        return ["--input-source", "body", "--backend", "sim", "--preview-body", *extra]
    raise AssertionError(f"unknown mode {mode}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="unirobot-infra-teleop-pico",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {teleop_module_version()}")
    subparsers = parser.add_subparsers(dest="mode", metavar="mode", required=True)
    for mode, help_text in _MODES:
        subparsers.add_parser(mode, help=help_text,
                              description=help_text + ". Additional teleop "
                              "arguments may follow, e.g. --duration, --headless, --record.")
    # Everything after the mode that this parser does not recognize is passed
    # through to the teleop core (argparse.REMAINDER does not swallow leading
    # "--" options in subparsers, so collect unknowns instead).
    ns, unknown = parser.parse_known_args(argv)
    extra = list(unknown)
    if ns.mode == "body":
        # Full Body 链懒加载，落位于 unirobot_infra.apps.teleop_pico_body。
        from unirobot_infra.apps import teleop_pico_body
        args = marvin_teleop.build_parser().parse_args(build_teleop_argv(ns.mode, extra, parser))
        teleop_pico_body.run(args)
        return
    marvin_teleop.main(build_teleop_argv(ns.mode, extra, parser))


def teleop_module_version():
    from importlib.metadata import version
    return version("unirobot-infra")


if __name__ == "__main__":
    main()
