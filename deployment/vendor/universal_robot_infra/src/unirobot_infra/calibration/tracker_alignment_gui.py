"""Interactive direction calibration. Reads tracking only; no robot connection."""

import time
from collections import deque

import numpy as np

from unirobot_infra.calibration.tracker_alignment import fit_alignment, save_alignment
from unirobot_infra.devices.pico_tracker import read_tracker_frame, tracker_timestamp


def calibrate(sdk, serials, path, frame_source=None):
    """frame_source: optional callable returning (poses, frame) like
    read_tracker_frame, e.g. the forearm-fused role stream in redundant mode."""
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("PICO 方向标定 — 前 / 左 / 上")
    root.geometry("850x650")
    root.minsize(750, 570)
    panel = ttk.Frame(root, padding=22)
    panel.pack(fill="both", expand=True)
    ttk.Label(panel, text="以你自己为准：X 向前 · Y 向左 · Z 向上",
              font=("Sans", 18, "bold")).pack(anchor="w")
    ttk.Label(panel, text="保持身体朝向固定，用左手做三个平移动作。右手尽量不动。\n"
              "每次从同一手位出发，移动 10–20 cm 后停住，点击采集；请同伴协助点击。",
              font=("Sans", 12)).pack(anchor="w", pady=12)
    instruction = tk.StringVar(value="1. 左手停在舒适的参考位置，点击『记录零点』。")
    ttk.Label(panel, textvariable=instruction, font=("Sans", 14, "bold"),
              wraplength=770).pack(anchor="w", pady=10)
    status = tk.StringVar(value="等待左手 tracker 数据…")
    ttk.Label(panel, textvariable=status, font=("Sans", 11)).pack(anchor="w")
    values = tk.StringVar(value="")
    ttk.Label(panel, textvariable=values, font=("Monospace", 12),
              justify="left").pack(anchor="w", pady=12)
    error = tk.StringVar(value="")
    ttk.Label(panel, textvariable=error, foreground="#bd3026", wraplength=770,
              font=("Sans", 11)).pack(anchor="w", pady=8)
    ttk.Label(panel, text="记录零点后，按顺序从该零点向前、向自己的左侧、向上移动。\n"
              "保存后，真机和仿真都会读取同一份方向标定。\n"
              "头显重新居中、追踪空间重建或身体换朝向后，应重新标定。",
              font=("Sans", 11)).pack(anchor="w", pady=8)

    samples = deque(maxlen=24)
    state = {"frame": None, "fresh": None, "origin": None, "strokes": [],
             "matrix": None, "errors": None, "result": None, "roles": {}}
    directions = ("向前", "向自己的左侧", "向上")

    def capture():
        error.set("")
        now = time.monotonic()
        recent = [p for t, p in samples if now - t < 0.4]
        if (state["fresh"] is None or now - state["fresh"] > 0.25
                or len(recent) < 5):
            error.set("左手 tracker 数据不完整或过期，请检查 Send 和左手序列号。")
            return
        points = np.array(recent)
        if np.max(np.linalg.norm(points - np.mean(points, axis=0), axis=1)) > 0.012:
            error.set("请保持左手静止约半秒再采集。")
            return
        point = np.mean(points, axis=0)
        if state["origin"] is None:
            state["origin"] = point
            button.configure(text="采集：向前")
            instruction.set("2. 左手从零点向前移动 10–20 cm，停住后点击『采集：向前』。")
            return
        stroke = point - state["origin"]
        if np.linalg.norm(stroke) < 0.08:
            error.set("移动距离不足 8 cm，请再移动一些。")
            return
        state["strokes"].append(stroke)
        index = len(state["strokes"])
        if index < 3:
            direction = directions[index]
            instruction.set(f"{index + 2}. 先回到原来的零点，再{direction}移动 10–20 cm，停住后采集。")
            button.configure(text=f"采集：{direction}")
        else:
            try:
                state["matrix"], state["errors"] = fit_alignment(state["strokes"])
            except ValueError as exc:
                state["strokes"].pop()
                error.set(str(exc) + " 若前两次动作有误，请点击『重新开始』。")
                return
            instruction.set("方向已算出。可以先看下方 X/Y/Z：向前应 +X、向左应 +Y、向上应 +Z。")
            button.configure(text="保存方向标定并继续", command=save)
            error.set("拟合角度误差（前 / 左 / 上）：" +
                      " / ".join(f"{v:.1f}°" for v in state["errors"]))

    def save():
        try:
            save_alignment(path, state["matrix"], serials, state["strokes"], state["errors"])
        except OSError as exc:
            error.set(f"保存失败：{exc}")
            return
        state["result"] = state["matrix"]
        root.destroy()

    def reset():
        state.update(origin=None, strokes=[], matrix=None, errors=None)
        samples.clear()
        instruction.set("1. 左手停在舒适的参考位置，点击『记录零点』。")
        error.set("")
        button.configure(text="记录零点", command=capture)

    buttons = ttk.Frame(panel)
    buttons.pack(anchor="w", pady=12)
    button = ttk.Button(buttons, text="记录零点", command=capture)
    button.pack(side="left", padx=(0, 10), ipadx=10, ipady=8)
    ttk.Button(buttons, text="重新开始", command=reset).pack(side="left", padx=10, ipady=8)
    ttk.Button(buttons, text="退出", command=root.destroy).pack(side="left", padx=10, ipady=8)

    def poll():
        try:
            if frame_source is not None:
                poses, frame = frame_source()
            else:
                poses, frame = read_tracker_frame(sdk)
            now = time.monotonic()
            if frame and frame != state["frame"]:
                state["frame"] = frame
                state["roles"] = {r: poses.get(sn) for r, sn in serials.items()}
                wrist = state["roles"]["left_wrist"]
                if wrist is not None:
                    if state["fresh"] is None:
                        clock, _ = tracker_timestamp(sdk)
                        print(f"Calibration receiving left wrist; valid trackers={len(poses)}, timestamp={clock}.", flush=True)
                    samples.append((now, wrist[:3].copy()))
                    state["fresh"] = now
            age = now - state["fresh"] if state["fresh"] is not None else float("inf")
            roles = ", ".join(r for r, p in state["roles"].items() if p is not None)
            clock, _ = tracker_timestamp(sdk)
            clock_label = "Motion 时间戳" if clock == "motion" else "总帧时间戳（兼容模式）"
            status.set(f"{'实时' if age < 0.25 else '等待 / 已断流'} | 数据 {age * 1000:.0f} ms | {clock_label}\n{roles}")
            if samples:
                point = samples[-1][1]
                text = f"左手原始位置 PICO (m): {np.round(point, 3)}\n"
                if state["origin"] is not None:
                    delta = point - state["origin"]
                    text += f"从零点位移 PICO (cm): {np.round(100 * delta, 1)}\n"
                    if state["matrix"] is not None:
                        text += f"你的 X前 / Y左 / Z上 (cm): {np.round(100 * state['matrix'] @ delta, 1)}"
                values.set(text)
        except Exception as exc:
            error.set(f"读取 tracker 失败：{exc}")
        root.after(15, poll)

    root.after(15, poll)
    root.mainloop()
    if state["result"] is None:
        raise RuntimeError("方向标定已取消，遥操作未启动。")
    return state["result"]
