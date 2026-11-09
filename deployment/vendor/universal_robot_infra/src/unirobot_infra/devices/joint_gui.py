from dataclasses import dataclass, field
import html
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import threading
from typing import Callable, Literal
from urllib.parse import urlparse

from omegaconf import DictConfig
from rich.console import Console

from unirobot_infra.core.time import now_ns
from unirobot_infra.core.types import DeviceFrame, RobotAction
from unirobot_infra.core.units import deg_to_rad, rad_to_deg


@dataclass(frozen=True)
class JointGuiJoint:
    name: str
    min_value: float
    max_value: float


@dataclass(frozen=True)
class JointGuiPart:
    side: str
    role: str
    label: str
    value_unit: Literal["deg", "rad"]
    step: float
    joints: tuple[JointGuiJoint, ...]


@dataclass
class JointGui:
    cfg: DictConfig
    parts: tuple[JointGuiPart, ...]
    name: str = "joint_gui"
    console: Console | None = None
    connected: bool = False
    on_action: Callable[[RobotAction], None] | None = None
    _q_rad: list[list[float]] = field(default_factory=list, init=False)
    _server: HTTPServer | None = field(default=None, init=False)
    _server_thread: threading.Thread | None = field(default=None, init=False)

    def connect(self) -> None:
        self._q_rad = [[0.0] * len(part.joints) for part in self.parts]
        self.connected = True

    def set_action_callback(self, callback: Callable[[RobotAction], None]) -> None:
        self.on_action = callback

    def set_initial_actions(self, actions: tuple[RobotAction, ...]) -> None:
        self._ensure_connected()
        action_by_side = {action.side: action for action in actions}
        for part_index, part in enumerate(self.parts):
            action = action_by_side[part.side]
            q_rad = self._action_part_q_rad(action, part)
            if len(q_rad) != len(part.joints):
                raise ValueError("JointGui initial action length mismatch")
            self._q_rad[part_index] = [float(value) for value in q_rad]

    def set_joint_value(self, part_index: int, joint_index: int, value: float) -> None:
        self._ensure_connected()
        self._validate_index(part_index, len(self.parts))
        self._validate_index(joint_index, len(self.parts[part_index].joints))
        self._q_rad[part_index][joint_index] = self._ui_to_rad(self.parts[part_index], float(value))
        self._emit_action(part_index)

    def read_frame(self) -> DeviceFrame:
        return self.read_side_frame(self._frame_side())

    def read_side_frame(self, side: str) -> DeviceFrame:
        self._ensure_connected()
        scalars: dict[str, float] = {}
        for part_index, part in enumerate(self.parts):
            if part.side != side:
                continue
            for joint_index, value in enumerate(self._q_rad[part_index]):
                scalars[f"{part.side}_{part.role}_q_rad_{joint_index}"] = value
        if not scalars:
            raise ValueError("JointGui side not found")
        return DeviceFrame(
            device_name=self.name,
            side=side,
            timestamp_ns=now_ns(),
            scalars=scalars,
        )

    def read_action(self) -> RobotAction:
        self._ensure_connected()
        side = self.parts[0].side
        for part in self.parts:
            if part.side != side:
                raise ValueError("JointGui read_action requires a single side")
        return self.read_side_action(side)

    def read_side_action(self, side: str) -> RobotAction:
        self._ensure_connected()
        arm_q_rad = None
        end_effector_q_rad = None
        for part_index, part in enumerate(self.parts):
            if part.side != side:
                continue
            q_rad = tuple(self._q_rad[part_index])
            if part.role == "arm":
                arm_q_rad = q_rad
            elif part.role == "end_effector":
                end_effector_q_rad = q_rad
            else:
                raise ValueError("unsupported JointGui part role")
        return RobotAction(
            side=side,
            timestamp_ns=now_ns(),
            arm_q_rad=arm_q_rad,
            end_effector_q_rad=end_effector_q_rad,
        )

    def run(self) -> None:
        self.start()
        if self._server_thread is None:
            raise RuntimeError("JointGui server thread is not running")
        self._server_thread.join()

    def start(self) -> None:
        self._ensure_connected()
        if self._server is not None:
            raise RuntimeError("JointGui server is already running")
        handler = self._web_handler()
        self._server = HTTPServer((str(self.cfg.host), int(self.cfg.port)), handler)
        self._print(f"Joint GUI: http://{self.cfg.host}:{self._server.server_port}/")
        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            name="JointGuiHTTPServer",
            daemon=True,
        )
        self._server_thread.start()

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        self._server_thread = None
        self.connected = False

    def _web_handler(self):
        gui = self

        class JointGuiRequestHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                path = urlparse(self.path).path
                if path == "/":
                    self._send_html(gui._web_html())
                    return
                if path == "/state":
                    self._send_json(gui._web_state())
                    return
                self.send_error(404)

            def do_POST(self) -> None:
                path = urlparse(self.path).path
                if path != "/joint":
                    self.send_error(404)
                    return
                content_length = int(self.headers["Content-Length"])
                result = gui._set_joint_from_web_json(self.rfile.read(content_length))
                self._send_json(result)

            def log_message(self, format: str, *args) -> None:
                return None

            def _send_html(self, body: str) -> None:
                data = body.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _send_json(self, body: dict) -> None:
                data = json.dumps(body).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        return JointGuiRequestHandler

    def _print(self, message: str) -> None:
        if self.console is not None:
            self.console.print(message)
            return
        print(message, flush=True)

    def _set_joint_from_web_json(self, data: bytes) -> dict:
        payload = json.loads(data.decode("utf-8"))
        self.set_joint_value(
            part_index=int(payload["part_index"]),
            joint_index=int(payload["joint_index"]),
            value=float(payload["value"]),
        )
        return self._web_state()

    def _web_state(self) -> dict:
        self._ensure_connected()
        return {
            "title": str(self.cfg.window_title),
            "parts": [
                {
                    "part_index": part_index,
                    "label": part.label,
                    "value_unit": part.value_unit,
                    "joints": [
                        {
                            "joint_index": joint_index,
                            "name": joint.name,
                            "min_value": joint.min_value,
                            "max_value": joint.max_value,
                            "step": part.step,
                            "value": self._rad_to_ui(part, self._q_rad[part_index][joint_index]),
                        }
                        for joint_index, joint in enumerate(part.joints)
                    ],
                }
                for part_index, part in enumerate(self.parts)
            ],
        }

    def _web_html(self) -> str:
        state_json = html.escape(json.dumps(self._web_state()), quote=False)
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(str(self.cfg.window_title))}</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f5f7fa;
      color: #18202a;
    }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: #f5f7fa;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 2;
      border-bottom: 1px solid #d7dde5;
      background: #ffffff;
      padding: 14px 24px;
    }}
    h1 {{
      margin: 0;
      font-size: 18px;
      font-weight: 650;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 20px 24px 32px;
    }}
    section {{
      margin-bottom: 24px;
    }}
    h2 {{
      margin: 0 0 10px;
      font-size: 15px;
      font-weight: 650;
    }}
    .joint-row {{
      display: grid;
      grid-template-columns: minmax(160px, 260px) minmax(220px, 1fr) 112px 52px;
      gap: 12px;
      align-items: center;
      min-height: 38px;
      border-top: 1px solid #dfe5ec;
    }}
    .joint-name {{
      overflow-wrap: anywhere;
      font-size: 13px;
    }}
    input[type="range"] {{
      width: 100%;
    }}
    input[type="number"] {{
      box-sizing: border-box;
      width: 112px;
      padding: 5px 7px;
      border: 1px solid #bac4cf;
      border-radius: 4px;
      background: #ffffff;
      color: #18202a;
      font: inherit;
      font-size: 13px;
    }}
    .unit {{
      color: #52606f;
      font-size: 12px;
    }}
    #status {{
      margin-top: 4px;
      color: #52606f;
      font-size: 12px;
    }}
    @media (max-width: 760px) {{
      header {{
        padding: 12px 14px;
      }}
      main {{
        padding: 16px 14px 24px;
      }}
      .joint-row {{
        grid-template-columns: 1fr;
        gap: 6px;
        padding: 10px 0;
      }}
      input[type="number"] {{
        width: 100%;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1></h1>
    <div id="status">ready</div>
  </header>
  <main id="app"></main>
  <script type="application/json" id="initial-state">{state_json}</script>
  <script>
    const app = document.querySelector("#app");
    const statusEl = document.querySelector("#status");
    const state = JSON.parse(document.querySelector("#initial-state").textContent);
    document.querySelector("h1").textContent = state.title;
    const timers = new Map();

    function build() {{
      app.replaceChildren();
      for (const part of state.parts) {{
        const section = document.createElement("section");
        const title = document.createElement("h2");
        title.textContent = part.label;
        section.appendChild(title);
        for (const joint of part.joints) {{
          section.appendChild(buildJointRow(part, joint));
        }}
        app.appendChild(section);
      }}
    }}

    function buildJointRow(part, joint) {{
      const row = document.createElement("div");
      row.className = "joint-row";

      const label = document.createElement("div");
      label.className = "joint-name";
      label.textContent = joint.name;

      const slider = document.createElement("input");
      slider.type = "range";
      slider.min = joint.min_value;
      slider.max = joint.max_value;
      slider.step = joint.step;
      slider.value = joint.value;

      const number = document.createElement("input");
      number.type = "number";
      number.min = joint.min_value;
      number.max = joint.max_value;
      number.step = joint.step;
      number.value = Number(joint.value).toFixed(4);

      const unit = document.createElement("span");
      unit.className = "unit";
      unit.textContent = part.value_unit;

      slider.addEventListener("input", () => {{
        number.value = Number(slider.value).toFixed(4);
        queueSend(part.part_index, joint.joint_index, slider.value);
      }});
      slider.addEventListener("change", () => sendJoint(part.part_index, joint.joint_index, slider.value));
      number.addEventListener("change", () => {{
        slider.value = number.value;
        sendJoint(part.part_index, joint.joint_index, number.value);
      }});

      row.append(label, slider, number, unit);
      return row;
    }}

    function queueSend(partIndex, jointIndex, value) {{
      const key = `${{partIndex}}:${{jointIndex}}`;
      clearTimeout(timers.get(key));
      timers.set(key, setTimeout(() => sendJoint(partIndex, jointIndex, value), 60));
    }}

    async function sendJoint(partIndex, jointIndex, value) {{
      statusEl.textContent = "sending";
      const response = await fetch("/joint", {{
        method: "POST",
        headers: {{"Content-Type": "application/json"}},
        body: JSON.stringify({{part_index: partIndex, joint_index: jointIndex, value: Number(value)}})
      }});
      if (!response.ok) {{
        statusEl.textContent = `error ${{response.status}}`;
        return;
      }}
      await response.json();
      statusEl.textContent = "ready";
    }}

    build();
  </script>
</body>
</html>
"""

    def _emit_action(self, part_index: int) -> None:
        if self.on_action is not None:
            self.on_action(self.read_side_action(self.parts[part_index].side))

    def _ensure_connected(self) -> None:
        if not self.connected:
            raise RuntimeError("JointGui is not connected")

    def _validate_index(self, index: int, size: int) -> None:
        if index < 0 or index >= size:
            raise ValueError("joint index out of range")

    def _frame_side(self) -> str:
        side = self.parts[0].side
        for part in self.parts:
            if part.side != side:
                raise ValueError("JointGui read_frame requires a single side")
        return side

    def _action_part_q_rad(self, action: RobotAction, part: JointGuiPart) -> tuple[float, ...]:
        if part.role == "arm":
            if action.arm_q_rad is None:
                raise ValueError("JointGui initial action missing arm_q_rad")
            return action.arm_q_rad
        if part.role == "end_effector":
            if action.end_effector_q_rad is None:
                raise ValueError("JointGui initial action missing end_effector_q_rad")
            return action.end_effector_q_rad
        raise ValueError("unsupported JointGui part role")

    def _rad_to_ui(self, part: JointGuiPart, value: float) -> float:
        if part.value_unit == "deg":
            return rad_to_deg(value)
        if part.value_unit == "rad":
            return value
        raise ValueError("unsupported JointGui value unit")

    def _ui_to_rad(self, part: JointGuiPart, value: float) -> float:
        if part.value_unit == "deg":
            return deg_to_rad(value)
        if part.value_unit == "rad":
            return value
        raise ValueError("unsupported JointGui value unit")


def build_joint_gui_parts(robot_cfg: DictConfig, gui_cfg: DictConfig) -> tuple[JointGuiPart, ...]:
    return tuple(
        _build_joint_gui_part(robot_cfg, side, gui_cfg, part_role)
        for side in robot_cfg.sides
        for part_role in robot_cfg.sides[side].parts
    )


def _build_joint_gui_part(
    robot_cfg: DictConfig,
    side: str,
    gui_cfg: DictConfig,
    part_role: str,
) -> JointGuiPart:
    if part_role == "arm":
        part_cfg = robot_cfg.hardware.arm
        return JointGuiPart(
            side=side,
            role=part_role,
            label=f"{side} arm",
            value_unit="deg",
            step=float(gui_cfg.arm_step_deg),
            joints=tuple(
                JointGuiJoint(
                    name=joint.name,
                    min_value=float(joint.min_deg),
                    max_value=float(joint.max_deg),
                )
                for joint in part_cfg.joints
            ),
        )
    if part_role == "end_effector":
        part_cfg = robot_cfg.hardware.end_effector
        return JointGuiPart(
            side=side,
            role=part_role,
            label=f"{side} end_effector",
            value_unit="rad",
            step=float(gui_cfg.end_effector_step_rad),
            joints=tuple(
                JointGuiJoint(
                    name=joint.name,
                    min_value=float(joint.min_rad),
                    max_value=float(joint.max_rad),
                )
                for joint in part_cfg.joints
            ),
        )
    raise ValueError("unsupported robot part role")
