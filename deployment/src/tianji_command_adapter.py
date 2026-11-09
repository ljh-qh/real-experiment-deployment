"""Project-local Tianji command transaction handling; targets are never altered."""
import logging
import math
import time

from unirobot_infra.robots.arm.tianji import TianjiRobot

logger = logging.getLogger(__name__)


class ReliableTianjiRobot(TianjiRobot):
    def _send_joint_positions_deg_with_retry(self, commands_by_arm):
        retries = int(self.cfg.send_cmd_retries)
        delay = float(self.cfg.send_cmd_retry_sleep_s)
        if retries < 0 or not math.isfinite(delay) or delay < 0:
            raise ValueError("Invalid Tianji command retry configuration")
        for arm_id, target in commands_by_arm.items():
            if arm_id not in ("A", "B") or len(target) != 7:
                raise ValueError("Expected arm A/B and seven joint targets")
            if not all(math.isfinite(value) for value in target):
                raise ValueError("Non-finite Tianji joint target")

        phase = "clear_set"
        for attempt in range(retries + 1):
            clear_ok = self._robot.clear_set()
            if clear_ok:
                for arm_id, target in commands_by_arm.items():
                    set_ok = self._robot.set_joint_cmd_pose(arm=arm_id, joints=list(target))
                    if not set_ok:
                        # False is not documented as retryable for this operation.
                        raise RuntimeError(
                            f"Tianji set_joint_cmd_pose failed for arm {arm_id}; "
                            f"attempt={attempt + 1}, clear_set={clear_ok!r}, "
                            f"set_result={set_ok!r}, target_deg={list(target)!r}"
                        )
                phase = "send_cmd"
                if self._robot.send_cmd():
                    return True
            else:
                phase = "clear_set"
            if attempt < retries:
                logger.warning(
                    "Tianji transaction retry phase=%s attempt=%d/%d clear_set=%r",
                    phase, attempt + 1, retries, clear_ok,
                )
                time.sleep(delay)
        raise RuntimeError(
            f"Tianji transaction failed at {phase} after {retries + 1} attempts; "
            f"clear_set={clear_ok!r}"
        )
