"""Unified command line entry for the MuJoCo Marvin teleop.

The main workflow uses four Object trackers with an explicit subcommand.
The separate 5-point Full Body path uses a headset, ankle trackers and
controllers::

    mujoco-marvin pico      --input-source body --backend sim   (Full Body)
    mujoco-marvin body      alias of pico
    mujoco-marvin preview   --input-source body --preview-body --backend sim
    mujoco-marvin sim       --backend sim (Object tracker input)
    mujoco-marvin calibrate --calibrate --backend sim (Object mode)
    mujoco-marvin arms      --backend robot   (needs a robot IP)
    mujoco-marvin both      --backend both    (needs a robot IP)

Any argument after the subcommand is passed through to the teleop core,
e.g. ``mujoco-marvin pico --demo --headless --duration 2``.

The robot IP resolves with the priority ``--robot-ip`` argument >
``TIANJI_ROBOT_IP`` environment variable > ``robot_ip`` in
``config/tianji_robot.yaml``; ``arms``/``both`` refuse to start without one.
"""

import argparse
import os

from . import teleop as teleop_module
from .robot_description import resolve_robot_config

_MODES = (
    ("pico", "PICO teleop of the sim via the separate 5-point Full Body skeleton"),
    ("body", "Alias of 'pico' (Full Body skeleton teleop of the sim model)"),
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
                    "set TIANJI_ROBOT_IP, or configure robot_ip in tianji_robot.yaml")
            argv += ["--robot-ip", ip]
        argv += ["--backend", "robot" if mode == "arms" else "both", *extra]
        return argv
    if mode in ("pico", "body"):
        return ["--input-source", "body", "--backend", "sim", *extra]
    if mode == "preview":
        return ["--input-source", "body", "--backend", "sim", "--preview-body", *extra]
    raise AssertionError(f"unknown mode {mode}")


def main(argv=None):
    # Launchers start this via `setsid nohup ... &`; non-interactive bash marks
    # background children SIGINT-ignored and CPython keeps that disposition,
    # so pkill -INT would silently do nothing. Restore KeyboardInterrupt.
    import signal
    if signal.getsignal(signal.SIGINT) == signal.SIG_IGN:
        signal.signal(signal.SIGINT, signal.default_int_handler)

    parser = argparse.ArgumentParser(
        prog="mujoco-marvin", description=__doc__,
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
    teleop_module.main(build_teleop_argv(ns.mode, extra, parser))


def teleop_module_version():
    from . import __version__
    return __version__


if __name__ == "__main__":
    main()
