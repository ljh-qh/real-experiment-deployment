"""Consume already-retargeted finger goals and dispatch one common hand command.

ROS is optional and imported only when --hand-input ros is selected. The
retargeter publishes /teleop/{side}_hand/joint_targets; this bridge publishes
hardware joint_commands only for robot/both and mirrors the same limits/rate
limited values in the simulation model.
"""

import time

import numpy as np


class HandCommandBridge:
    def __init__(self, controller, hardware=False, timeout=.25, speed_deg_s=90.):
        import rclpy
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        from sensor_msgs.msg import JointState
        self.rclpy = rclpy
        self.message_type = JointState
        self.owns_context = not rclpy.ok()
        if self.owns_context:
            rclpy.init(args=[])
        self.node = rclpy.create_node("pico_body_hand_bridge")
        self.controller = controller
        self.timeout = timeout
        self.max_speed = np.deg2rad(speed_deg_s)
        self.hardware = hardware
        self.goals = {}
        self.seen = {}
        self.feedback_stamp = {}
        self.feedback = {}
        self.last_stamp = {}
        self.requires_fresh_after = time.monotonic()
        self.was_enabled = False
        self.ids = {}
        self.limits = {}
        self.names = {}
        self.publishers = {}
        self.ready = {side: not hardware for side in ("left", "right")}
        self.state = {side: "WAITING" for side in ("left", "right")}
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        for side in ("left", "right"):
            names = [f"{side}_finger{finger}_joint{joint}" for finger in range(1,6) for joint in range(1,5)]
            joints = [controller.model.joint(name).id for name in names]
            self.names[side] = names
            self.ids[side] = np.array([controller.model.jnt_qposadr[j] for j in joints])
            self.limits[side] = controller.model.jnt_range[joints].copy()
            self.node.create_subscription(JointState, f"/teleop/{side}_hand/joint_targets",
                                          lambda msg, s=side: self._target(s, msg), qos)
            if hardware:
                self.node.create_subscription(JointState, f"/{side}_hand/joint_states",
                                              lambda msg, s=side: self._feedback(s, msg), qos)
                self.publishers[side] = self.node.create_publisher(JointState, f"/{side}_hand/joint_commands", qos)

    def _positions(self, side, msg):
        values = np.asarray(msg.position, dtype=float)
        if values.shape != (20,) or not np.isfinite(values).all():
            raise ValueError("Expected 20 finite finger angles")
        if msg.name:
            if len(msg.name) != 20 or set(msg.name) != set(self.names[side]):
                raise ValueError("Finger joint names do not match URDF")
            lookup = dict(zip(msg.name, values))
            values = np.array([lookup[name] for name in self.names[side]])
        return values

    def _target(self, side, msg):
        try:
            values = self._positions(side, msg)
            stamp = msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec
            clock_now = self.node.get_clock().now().nanoseconds
            if stamp <= 0 or abs(clock_now - stamp) > int(self.timeout * 1e9):
                self.state[side] = "STALE TARGET"
                return
            if stamp <= self.last_stamp.get(side, 0):
                return
            self.last_stamp[side] = stamp
            self.goals[side] = np.clip(values, *self.limits[side].T)
            self.seen[side] = time.monotonic()
        except ValueError as exc:
            self.state[side] = str(exc)

    def _feedback(self, side, msg):
        try:
            stamp = msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec
            if stamp <= 0 or abs(self.node.get_clock().now().nanoseconds - stamp) > int(self.timeout*1e9):
                self.state[side] = "STALE FEEDBACK"
                return
            if stamp <= self.feedback_stamp.get(side, 0):
                return
            values = self._positions(side, msg)
            self.feedback_stamp[side] = stamp
            self.feedback[side] = (values, time.monotonic())
        except ValueError:
            self.state[side] = "INVALID FEEDBACK"

    def tick(self, enabled, dt, output_active=True):
        # Queue depth is one; bound callback processing to keep IK responsive.
        for _ in range(4):
            self.rclpy.spin_once(self.node, timeout_sec=0.)
        now = time.monotonic()
        if enabled and not self.was_enabled:
            self.requires_fresh_after = now
        self.was_enabled = enabled
        for side in ("left", "right"):
            ids = self.ids[side]
            if self.hardware and not self.ready[side]:
                feedback = self.feedback.get(side)
                if feedback is None or now - feedback[1] > self.timeout:
                    self.state[side] = "WAITING FEEDBACK"
                    continue
                if np.any(feedback[0] < self.limits[side][:,0]-.01) or np.any(feedback[0] > self.limits[side][:,1]+.01):
                    self.state[side] = "FEEDBACK OUTSIDE MODEL LIMITS"
                    continue
                self.controller.command[ids] = np.clip(feedback[0], *self.limits[side].T)
                self.ready[side] = True
            if self.hardware:
                feedback = self.feedback.get(side)
                if feedback is None or now - feedback[1] > self.timeout:
                    self.ready[side] = False
                    self.state[side] = "FEEDBACK LOST"
                    continue
            fresh = side in self.seen and now - self.seen[side] <= self.timeout
            if enabled and fresh and self.seen[side] > self.requires_fresh_after:
                delta = self.goals[side] - self.controller.command[ids]
                self.controller.command[ids] += np.clip(delta, -self.max_speed * dt, self.max_speed * dt)
                self.state[side] = "FOLLOWING"
            else:
                self.state[side] = "HOLD" if fresh else "NO FRESH FINGER TARGET"
            if self.hardware and output_active:
                msg = self.message_type()
                msg.header.stamp = self.node.get_clock().now().to_msg()
                msg.position = self.controller.command[ids].tolist()
                self.publishers[side].publish(msg)
        return {s: self.controller.command[ids].tolist() for s, ids in self.ids.items()}

    def close(self):
        self.node.destroy_node()
        if self.owns_context:
            self.rclpy.shutdown()
