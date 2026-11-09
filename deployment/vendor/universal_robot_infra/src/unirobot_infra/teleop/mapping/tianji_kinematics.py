from pathlib import Path
import importlib
import logging
import sys
from collections.abc import Sequence
from typing import Any

from omegaconf import DictConfig

from unirobot_infra.robots.arm.tianji import sdk_arm_to_index


def ensure_sdk_dir_on_path(sdk_dir: str) -> None:
    sdk_dir_path = Path(sdk_dir)
    if not sdk_dir_path.exists():
        raise FileNotFoundError(str(sdk_dir_path))
    sdk_dir_str = str(sdk_dir_path)
    if sdk_dir_str not in sys.path:
        sys.path.insert(0, sdk_dir_str)


def import_fx_kine_module(sdk_dir: str):
    ensure_sdk_dir_on_path(sdk_dir)
    return importlib.import_module("fx_kine")


def kine_cfg_path(cfg: DictConfig) -> Path:
    path = Path(cfg.kine_cfg)
    if path.is_absolute():
        return path
    return Path(cfg.sdk_dir) / path


def quiet_fx_kine_logger() -> None:
    logging.getLogger("debug_printer").setLevel(logging.ERROR)


class TianjiKinematics:
    def __init__(self, cfg: DictConfig, arm_id: str) -> None:
        self.cfg = cfg
        self.arm_id = arm_id.strip().upper()
        self.arm_index = sdk_arm_to_index(self.arm_id)
        self._kine: Any = None
        self.initialized = False

    def initialize(self) -> None:
        quiet_fx_kine_logger()
        fx_kine = import_fx_kine_module(self.cfg.sdk_dir)
        kine = fx_kine.Marvin_Kine()
        kine.log_switch(0)

        cfg_path = kine_cfg_path(self.cfg)
        cfg_data = kine.load_config(self.arm_index, str(cfg_path))
        if not cfg_data:
            raise RuntimeError(f"Failed to load Tianji kinematics config: {cfg_path}")

        robot_type = int(cfg_data["TYPE"][self.arm_index])
        dh = cfg_data["DH"][self.arm_index]
        pnva = cfg_data["PNVA"][self.arm_index]
        j67 = cfg_data["BD"][self.arm_index]
        if not kine.initial_kine(robot_type=robot_type, dh=dh, pnva=pnva, j67=j67):
            raise RuntimeError(f"Failed to initialize Tianji kinematics for arm {self.arm_id}")

        self._kine = kine
        self.initialized = True

    def fk_xyzabc_mm_deg(self, q_deg: Sequence[float]) -> tuple[float, float, float, float, float, float]:
        self._ensure_initialized()
        q = _to_vec7("q_deg", q_deg)
        fk_mat = self._kine.fk(list(q))
        if not fk_mat:
            raise RuntimeError("Tianji FK failed")
        xyzabc = self._kine.mat4x4_to_xyzabc(fk_mat)
        if not xyzabc:
            raise RuntimeError("Tianji mat4x4_to_xyzabc failed")
        if len(xyzabc) < 6:
            raise RuntimeError("Tianji FK output xyzabc must have at least 6 values")
        return (
            float(xyzabc[0]),
            float(xyzabc[1]),
            float(xyzabc[2]),
            float(xyzabc[3]),
            float(xyzabc[4]),
            float(xyzabc[5]),
        )

    def xyzabc_to_mat4x4(
        self,
        xyzabc_mm_deg: Sequence[float],
    ) -> tuple[tuple[float, float, float, float], ...]:
        self._ensure_initialized()
        target = _to_vec6("xyzabc_mm_deg", xyzabc_mm_deg)
        pose_mat = self._kine.xyzabc_to_mat4x4(list(target))
        if not pose_mat:
            raise RuntimeError("Tianji xyzabc_to_mat4x4 failed")
        return _to_mat4x4("pose_mat", pose_mat)

    def mat4x4_to_xyzabc_mm_deg(
        self,
        pose_mat: Sequence[Sequence[float]],
    ) -> tuple[float, float, float, float, float, float]:
        self._ensure_initialized()
        mat = _to_mat4x4("pose_mat", pose_mat)
        xyzabc = self._kine.mat4x4_to_xyzabc([list(row) for row in mat])
        if not xyzabc:
            raise RuntimeError("Tianji mat4x4_to_xyzabc failed")
        if len(xyzabc) < 6:
            raise RuntimeError("Tianji mat4x4_to_xyzabc output must have at least 6 values")
        return (
            float(xyzabc[0]),
            float(xyzabc[1]),
            float(xyzabc[2]),
            float(xyzabc[3]),
            float(xyzabc[4]),
            float(xyzabc[5]),
        )

    def ik_xyzabc_mm_deg(
        self,
        xyzabc_mm_deg: Sequence[float],
        reference_q_deg: Sequence[float],
    ) -> tuple[float, float, float, float, float, float, float]:
        self._ensure_initialized()
        target = _to_vec6("xyzabc_mm_deg", xyzabc_mm_deg)
        reference = _to_vec7("reference_q_deg", reference_q_deg)

        fx_kine = import_fx_kine_module(self.cfg.sdk_dir)

        pose_mat = self._kine.xyzabc_to_mat4x4(list(target))
        if not pose_mat:
            raise RuntimeError("Tianji xyzabc_to_mat4x4 failed")
        pose_1x16 = self._kine.mat4x4_to_mat1x16(pose_mat)
        if not pose_1x16 or len(pose_1x16) != 16:
            raise RuntimeError("Tianji mat4x4_to_mat1x16 failed")

        solve_para = fx_kine.FX_InvKineSolvePara()
        solve_para.set_input_ik_target_tcp(pose_1x16)
        solve_para.set_input_ik_ref_joint(reference)
        solve_para.set_input_ik_zsp_type(int(self.cfg.ik.zsp_type))
        solve_para.set_input_ik_zsp_para([float(value) for value in self.cfg.ik.zsp_para])

        result = self._kine.ik(structure_data=solve_para)
        if not result:
            raise RuntimeError("Tianji IK failed")
        if bool(result.m_Output_IsOutRange):
            raise RuntimeError("Tianji IK target is out of range")
        if bool(result.m_Output_IsJntExd):
            raise RuntimeError("Tianji IK joint result exceeds limits")
        q_deg = result.m_Output_RetJoint.to_list()
        return _to_vec7("ik_result_q_deg", q_deg)

    def _ensure_initialized(self) -> None:
        if not self.initialized or self._kine is None:
            raise RuntimeError("TianjiKinematics is not initialized")


def _to_vec6(name: str, values: Sequence[float]) -> tuple[float, float, float, float, float, float]:
    output = tuple(float(value) for value in values)
    if len(output) != 6:
        raise ValueError(f"{name} must have length 6")
    return (output[0], output[1], output[2], output[3], output[4], output[5])


def _to_vec7(name: str, values: Sequence[float]) -> tuple[float, float, float, float, float, float, float]:
    output = tuple(float(value) for value in values)
    if len(output) != 7:
        raise ValueError(f"{name} must have length 7")
    return (output[0], output[1], output[2], output[3], output[4], output[5], output[6])


def _to_mat4x4(name: str, values: Sequence[Sequence[float]]) -> tuple[tuple[float, float, float, float], ...]:
    rows = tuple(tuple(float(value) for value in row) for row in values)
    if len(rows) != 4:
        raise ValueError(f"{name} must have 4 rows")
    if any(len(row) != 4 for row in rows):
        raise ValueError(f"{name} must have 4 columns")
    return (
        (rows[0][0], rows[0][1], rows[0][2], rows[0][3]),
        (rows[1][0], rows[1][1], rows[1][2], rows[1][3]),
        (rows[2][0], rows[2][1], rows[2][2], rows[2][3]),
        (rows[3][0], rows[3][1], rows[3][2], rows[3][3]),
    )
