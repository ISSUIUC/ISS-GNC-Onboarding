"""Behaviour checker for notebooks/orientation_estimation.ipynb."""

from __future__ import annotations

import numpy as np


def _to_quat_vec(q) -> np.ndarray | None:
    """Safely convert a quaternion object or array to [w, x, y, z]."""
    if q is None:
        return None
    if hasattr(q, "w") and hasattr(q, "x") and hasattr(q, "y") and hasattr(q, "z"):
        return np.array([float(q.w), float(q.x), float(q.y), float(q.z)])
    if hasattr(q, "components"):
        return np.array(q.components, dtype=float)
    if isinstance(q, (list, tuple, np.ndarray)) and len(q) == 4:
        return np.array(q, dtype=float)
    return None


def _quat_close(q1, q2, atol: float = 1e-3) -> bool:
    v1 = _to_quat_vec(q1)
    v2 = _to_quat_vec(q2)
    if v1 is None or v2 is None:
        return False
    return bool(np.allclose(v1, v2, atol=atol) or np.allclose(v1, -v2, atol=atol))


def check_ex1(ctx):
    """Grades orientation_estimation-ex1: Euler Rotation Matrices."""
    yaw = np.deg2rad(30)
    pitch = np.deg2rad(20)
    roll = np.deg2rad(45)

    ref_Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,            0,           1]
    ])
    ref_Ry = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [ 0,             1, 0            ],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    ref_Rx = np.array([
        [1, 0,           0          ],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])
    ref_R = ref_Rz @ ref_Ry @ ref_Rx

    R = ctx.get("R")
    R_x = ctx.get("R_x")
    R_y = ctx.get("R_y")
    R_z = ctx.get("R_z")

    if not ctx.require("Defines total rotation matrix R", R is not None, "Assign your final combined rotation matrix to `R`."):
        return

    ctx.require("R has shape (3, 3)", isinstance(R, np.ndarray) and R.shape == (3, 3), f"R must be a 3x3 matrix (got shape {getattr(R, 'shape', type(R))}).")
    
    if R_x is not None:
        ctx.require("R_x matrix is correct for roll=45°", isinstance(R_x, np.ndarray) and R_x.shape == (3, 3) and np.allclose(R_x, ref_Rx, atol=1e-3), "Check your formula for R_x (rotation around x-axis).")
    if R_y is not None:
        ctx.require("R_y matrix is correct for pitch=20°", isinstance(R_y, np.ndarray) and R_y.shape == (3, 3) and np.allclose(R_y, ref_Ry, atol=1e-3), "Check your formula for R_y (rotation around y-axis).")
    if R_z is not None:
        ctx.require("R_z matrix is correct for yaw=30°", isinstance(R_z, np.ndarray) and R_z.shape == (3, 3) and np.allclose(R_z, ref_Rz, atol=1e-3), "Check your formula for R_z (rotation around z-axis).")

    ctx.require("Combined matrix R matches R_z @ R_y @ R_x", isinstance(R, np.ndarray) and R.shape == (3, 3) and np.allclose(R, ref_R, atol=1e-3), "Ensure you multiply matrices in order: R = R_z @ R_y @ R_x.")


def check_ex2(ctx):
    """Grades orientation_estimation-ex2: Rotation Quaternions."""
    import quaternion as q_lib

    yaw = np.deg2rad(30)
    pitch = np.deg2rad(20)
    roll = np.deg2rad(45)

    ref_qz = q_lib.quaternion(np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
    ref_qy = q_lib.quaternion(np.cos(pitch / 2), 0, np.sin(pitch / 2), 0)
    ref_qx = q_lib.quaternion(np.cos(roll / 2), np.sin(roll / 2), 0, 0)
    ref_quat = ref_qz * ref_qy * ref_qx

    quat = ctx.get("quat", "q", "q_total")
    q_z = ctx.get("q_z")
    q_y = ctx.get("q_y")
    q_x = ctx.get("q_x")

    if not ctx.require("Defines combined rotation quaternion `quat`", quat is not None, "Assign your combined quaternion to `quat`."):
        return

    if q_z is not None:
        ctx.require("q_z quaternion is correct for yaw=30°", _quat_close(q_z, ref_qz), "Check q_z formula: (cos(yaw/2), 0, 0, sin(yaw/2)).")
    if q_y is not None:
        ctx.require("q_y quaternion is correct for pitch=20°", _quat_close(q_y, ref_qy), "Check q_y formula: (cos(pitch/2), 0, sin(pitch/2), 0).")
    if q_x is not None:
        ctx.require("q_x quaternion is correct for roll=45°", _quat_close(q_x, ref_qx), "Check q_x formula: (cos(roll/2), sin(roll/2), 0, 0).")

    ctx.require("Combined quaternion quat matches q_z * q_y * q_x", _quat_close(quat, ref_quat), "Multiply quaternions: quat = q_z * q_y * q_x.")

    v = _to_quat_vec(quat)
    if v is not None:
        norm = float(np.linalg.norm(v))
        ctx.require("Combined quaternion is unit quaternion (norm ≈ 1)", np.isclose(norm, 1.0, atol=1e-3), f"Quaternion norm is {norm:.4f}, expected 1.0.")
