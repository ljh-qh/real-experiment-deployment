"""Tianji closed-form IK (open-source fx_kine SDK) with strict flag semantics.

The vendor module is loaded by file path — the same precedent as
``robots/arm/tianji_marvin_backend.load_sdk``: the open-source SDK repo keeps
``fx_kine.py`` beside its ``libKine.so`` under ``SDK_PYTHON/`` and needs no
package install.  Importing this module never requires the SDK; constructing
:class:`SdkKinematics` does.

SDK directory resolution priority: ``sdk_dir argument > TJ_FX_SDK_DIR >
MARVIN_SDK_DIR > ./vendor``.  The arm config (``.MvKDCfg``) resolution:
``kine_config argument > MARVIN_KINE_CFG > $SDK_DIR/CommonConfig/ccs_m6_40.MvKDCfg``.

Cross-solver convention (see docs/ik_example_comparison.md): the SDK model and
the MuJoCo URDF agree on joint order/sign/zero but not on link translation, so
SDK targets are always produced by the SDK's own FK (self-consistent), never by
feeding MuJoCo TCP translations into the SDK.
"""

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SDK_HELP = (
    "Tianji FX kinematics SDK not found. Clone the open-source SDK repo: "
    "git clone --depth 1 -b master https://gh-proxy.com/https://github.com/cynthia-you/TJ_FX_ROBOT_CONTRL_SDK /tmp/tj_fx_sdk "
    "then set TJ_FX_SDK_DIR (or MARVIN_SDK_DIR) to it, or pass sdk_dir. "
    "Only the --pose move (G key) and the arm-angle plan bias (B key) need it; "
    "pure teleoperation does not."
)

_DEFAULT_KINE_CONFIG = Path("CommonConfig") / "ccs_m6_40.MvKDCfg"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_sdk_dir(sdk_dir=None):
    """Resolve the open-source FX SDK repo root (argument > env > ./vendor)."""
    candidate = Path(sdk_dir or os.environ.get("TJ_FX_SDK_DIR")
                     or os.environ.get("MARVIN_SDK_DIR") or Path.cwd() / "vendor")
    if not (candidate / "SDK_PYTHON" / "fx_kine.py").is_file():
        raise RuntimeError(f"{_SDK_HELP} (looked in {candidate})")
    return candidate


def resolve_kine_config(kine_config=None, sdk_dir=None):
    """Resolve the arm .MvKDCfg (argument > MARVIN_KINE_CFG > SDK default)."""
    candidate = Path(kine_config or os.environ.get("MARVIN_KINE_CFG")
                     or Path(sdk_dir) / _DEFAULT_KINE_CONFIG)
    if not candidate.is_file():
        raise FileNotFoundError(f"Kine config not found: {candidate}")
    return candidate


@dataclass
class SdkIkResult:
    """Strict reading of the SDK IK output flags; no guessing on failure."""
    q7_deg: tuple
    ok: bool
    out_of_range: bool
    joint_exceeded: bool
    degenerate: bool
    exceeded_tags: tuple
    branch_count: int


class SdkKinematics:
    """Closed-form FK/IK for one arm through the vendor ``fx_kine`` module.

    ``arm_type`` 0 is the left arm, 1 the right arm (vendor convention).
    ``solve_pose`` runs the mandatory ``ik()`` first; ``zsp_angle`` adds the
    ``ik_nsp()`` arm-angle pass on the same TCP.  With ``zsp_type=1`` the
    reference plane is ``fk_nsp(ref)`` first column, as in the vendor demo.
    """

    def __init__(self, sdk_dir=None, kine_config=None, arm_type=0):
        if arm_type not in (0, 1):
            raise ValueError("arm_type must be 0 (left) or 1 (right)")
        self.arm_type = arm_type
        directory = resolve_sdk_dir(sdk_dir)
        config = resolve_kine_config(kine_config, directory)
        self._fx_kine = _load_module("tj_fx_kine", directory / "SDK_PYTHON" / "fx_kine.py")
        self._kk = self._fx_kine.Marvin_Kine()
        self._kk.log_switch(0)  # silence the C library's stdout banner
        ini = self._kk.load_config(arm_type=arm_type, config_path=str(config))
        if ini is None:
            raise RuntimeError(f"libKine failed to load config {config}")
        if not self._kk.initial_kine(robot_type=ini["TYPE"][arm_type], dh=ini["DH"][arm_type],
                                     pnva=ini["PNVA"][arm_type], j67=ini["BD"][arm_type]):
            raise RuntimeError(f"libKine initial_kine failed for arm_type={arm_type} with {config}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def fk_q7(self, q7_deg):
        """Forward kinematics of one configuration -> 4x4 pose, translation in mm."""
        q = np.asarray(q7_deg, dtype=float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError("q7_deg must be seven finite angles in degrees")
        mat = self._kk.fk(joints=q.tolist())
        if not mat:
            raise RuntimeError("libKine FK failed")
        return np.asarray(mat, dtype=float).reshape(4, 4)

    def current_xyzabc(self, q7_deg):
        """FK shortcut: joint angles (deg) -> xyzabc (mm, deg) in the arm base frame."""
        xyzabc = self._kk.mat4x4_to_xyzabc(self.fk_q7(q7_deg).tolist())
        if not xyzabc:
            raise RuntimeError("libKine mat4x4_to_xyzabc failed")
        return np.asarray(xyzabc, dtype=float)

    def solve_pose(self, xyzabc_mm_deg, ref_q7_deg, zsp_type=0, zsp_angle=None):
        """Closed-form IK for one absolute pose near the reference configuration.

        ``xyzabc_mm_deg`` is in the SDK arm base frame (mm/deg); the solution
        stays on the reference branch of ``ref_q7_deg``.  When ``zsp_angle`` is
        given, ``ik_nsp`` rotates the arm angle by that many degrees on the
        unchanged TCP (requires a prior ``ik()`` on the same struct, which this
        method performs).  Failure is reported through the result flags exactly
        as the SDK sets them.
        """
        xyzabc = np.asarray(xyzabc_mm_deg, dtype=float)
        ref = np.asarray(ref_q7_deg, dtype=float)
        if xyzabc.shape != (6,) or not np.isfinite(xyzabc).all():
            raise ValueError("xyzabc_mm_deg must be six finite numbers (mm, deg)")
        if ref.shape != (7,) or not np.isfinite(ref).all():
            raise ValueError("ref_q7_deg must be seven finite angles in degrees")
        kk = self._kk
        sp = self._fx_kine.FX_InvKineSolvePara()
        sp.set_input_ik_target_tcp(kk.mat4x4_to_mat1x16(kk.xyzabc_to_mat4x4(xyzabc.tolist())))
        sp.set_input_ik_ref_joint(ref.tolist())
        sp.set_input_ik_zsp_type(zsp_type)
        if zsp_type == 1:
            _, nsp = kk.fk_nsp(ref.tolist())
            if not nsp:
                raise RuntimeError("libKine FK_NSP failed for the reference plane")
            sp.set_input_ik_zsp_para([nsp[0][0], nsp[1][0], nsp[2][0], 0., 0., 0.])
        ok = bool(kk.ik(structure_data=sp))
        if ok and zsp_angle is not None:
            sp.set_input_zsp_angle(float(zsp_angle))
            ok = bool(kk.ik_nsp(sturcture_data=sp))
        return self._result(sp, ok)

    @staticmethod
    def _result(sp, ok):
        joints = sp.m_Output_RetJoint.to_list()
        if len(joints) != 7:
            raise ValueError(f"SDK returned {len(joints)} joints, expected 7")
        return SdkIkResult(
            q7_deg=tuple(float(v) for v in joints),
            ok=ok,
            out_of_range=bool(sp.m_Output_IsOutRange),
            joint_exceeded=bool(sp.m_Output_IsJntExd),
            degenerate=any(bool(d) for d in sp.m_Output_IsDeg),
            exceeded_tags=tuple(bool(t) for t in sp.m_Output_JntExdTags),
            branch_count=int(sp.m_OutPut_Result_Num),
        )
