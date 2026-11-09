#!/usr/bin/env python3
"""Plot a teleop JSONL recording using NumPy and Matplotlib (no robot SDK).

Requires the ``plot`` extra (matplotlib). Run as
``python -m mujoco_marvin.plot_recording recording.jsonl --output plot.png``.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def constant_position_spans(poses, times, minimum=.5):
    spans = []
    begin = 0
    for i in range(1, len(poses) + 1):
        if i < len(poses) and np.array_equal(poses[i, :3], poses[begin, :3]):
            continue
        if times[i - 1] - times[begin] >= minimum:
            spans.append((times[begin], times[i - 1]))
        begin = i
    return spans


def plot_recording(path, output):
    records = [json.loads(line) for line in Path(path).read_text().splitlines()]
    metadata = records[0]
    rows = [r for r in records[1:] if r.get("arms")]
    if not rows:
        raise ValueError("Recording contains no live tracker reference")
    times = np.array([r["elapsed_s"] for r in rows])
    times -= times[0]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(5, 2, figsize=(14, 14), sharex=True, constrained_layout=True)
    fig.suptitle("30-second teleoperation: position input holds while rotation keeps updating\n"
                 "Time starts when the tracker reference is captured; shaded = identical input XYZ",
                 fontsize=15)
    spans_by_role = {}
    for column, side in enumerate(("left", "right")):
        serial = metadata["tracker_serials"][f"{side}_wrist"]
        poses = np.array([r["poses"][serial] for r in rows])
        spans = constant_position_spans(poses, times)
        spans_by_role[f"{side}_wrist"] = spans
        for axis, color in enumerate(("#cc4141", "#158958", "#2768bc")):
            axes[0, column].plot(times, 100 * (poses[:, axis] - poses[0, axis]),
                                 color=color, label=f"raw {'XYZ'[axis]}", lw=1.3)
        axes[0, column].set_title(f"{side.capitalize()} wrist [{serial[-5:]}]")
        axes[0, column].set_ylabel("Received position delta (cm)")
        axes[0, column].legend(loc="upper left", ncol=3)
        for component in range(4):
            axes[1, column].plot(times, poses[:, 3 + component],
                                 label=f"q{'xyzw'[component]}", lw=1.)
        axes[1, column].set_ylabel("Received quaternion")
        axes[1, column].legend(loc="upper left", ncol=4)
        arm = [r["arms"][side] for r in rows]
        axes[2, column].plot(times, [1000 * a["position_error_m"] for a in arm],
                             color="#c93b3b", label="Position error (mm)")
        axes[2, column].plot(times, [a["rotation_error_deg"] for a in arm],
                             color="#7a4db6", label="Rotation error (deg)")
        axes[2, column].set_ylabel("IK goal minus model")
        axes[2, column].legend(loc="upper left")
        margin = np.array([a["joint_margin_deg"] for a in arm])
        axes[3, column].plot(times, margin.min(axis=1), label="Minimum joint margin", color="#087e8b")
        axes[3, column].axhline(1., color="#c93b3b", ls="--", lw=.8, label="1 degree")
        axes[3, column].set_ylabel("Joint limit margin (deg)")
        axes[3, column].legend(loc="upper left")
        for row in range(4):
            for begin, end in spans:
                axes[row, column].axvspan(begin, end, color="#eead47", alpha=.17, zorder=-1)

    for index, role in enumerate(metadata["tracker_serials"]):
        serial = metadata["tracker_serials"][role]
        poses = np.array([r["poses"][serial] for r in rows])
        spans_by_role[role] = constant_position_spans(poses, times)
        for begin, end in spans_by_role[role]:
            axes[4, 0].broken_barh([(begin, end - begin)], (index - .28, .56),
                                   facecolors="#df9b2d")
    axes[4, 0].set_yticks(range(4), list(metadata["tracker_serials"]))
    axes[4, 0].set_title("Intervals of byte-identical XYZ (at least 0.5 s)")
    axes[4, 0].set_ylim(-.5, 3.5)
    axes[4, 1].axis("off")
    longest = max(spans_by_role["left_wrist"], key=lambda span: span[1] - span[0], default=None)
    observations = ["Position and rotation are received from the same seven-value pose."]
    if longest is not None:
        begin, end = longest
        selected = [r for r, t in zip(rows, times) if begin <= t <= end]
        wrist = metadata["tracker_serials"]["left_wrist"]
        quats = np.array([r["poses"][wrist][3:] for r in selected])
        changes = np.count_nonzero(np.any(np.diff(quats, axis=0) != 0, axis=1))
        worst_error = max(r["arms"]["left"]["position_error_m"] for r in selected) * 1000
        margin = min(min(r["arms"]["left"]["joint_margin_deg"]) for r in selected)
        observations.extend([
            f"Left wrist: XYZ repeats for {end - begin:.2f} s; quaternion changes {changes} times.",
            f"In that interval: position error <= {worst_error:.3f} mm, joint margin >= {margin:.1f} deg.",
        ])
    for role in ("left_arm", "right_arm"):
        spans = spans_by_role[role]
        longest_duration = max((end - begin for begin, end in spans), default=0.)
        observations.append(f"{role}: longest identical XYZ interval = {longest_duration:.2f} s.")
    axes[4, 1].text(.03, .93,
                   "Key observations\n\n"
                   + "\n".join(observations) + "\n\n"
                   "Unchanged input is not proof of the hardware cause.\n"
                   "A per-device position-tracking validity flag was not recorded.",
                   transform=axes[4, 1].transAxes, va="top", fontsize=10)
    for row in axes:
        for ax in row:
            ax.set_xlim(0, 30)
            ax.grid(alpha=.17)
    for ax in axes[-1]:
        ax.set_xlabel("Operation time (s)")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    plot_recording(args.recording, args.output)


if __name__ == "__main__":
    main()
