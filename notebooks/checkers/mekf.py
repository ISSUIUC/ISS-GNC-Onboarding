"""Behaviour checkers for MEKF.ipynb.

Three exercises covering the Multiplicative Extended Kalman Filter (MEKF):
1. check_ex1: Skew-symmetric matrix function `skew(v)`
2. check_ex2: MEKF state prediction function `predict_state`
3. check_ex3: MEKF measurement update function `update_state`
"""

import numpy as np
import quaternion as qt


def _ran(ctx, hint="Fix the error shown in the output above, then check again."):
    """Every exercise starts here: nothing else means anything if it crashed."""
    return ctx.require("Runs without an error", ctx.ok, hint)


def _number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


# ---------------------------------------------------------------------------
# Exercise 1 — skew-symmetric matrix
# ---------------------------------------------------------------------------

def check_ex1(ctx):
    """Verify `skew(v)` maps a 3D vector to a 3x3 skew-symmetric matrix."""
    if not _ran(ctx):
        return

    skew_func = ctx.env.get("skew")
    if not ctx.require("Defines a function named `skew`", callable(skew_func),
                       "Define a function `skew(v)` that returns a 3x3 numpy matrix."):
        return

    # Test with sample vectors
    test_vectors = [
        np.array([1.0, 2.0, 3.0]),
        np.array([0.0, -4.5, 2.1]),
        np.array([-1.2, 0.0, 5.5])
    ]

    for v in test_vectors:
        expected = np.array([
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0]
        ])
        try:
            got = skew_func(v)
        except Exception as e:
            ctx.require("`skew(v)` runs without throwing an exception", False,
                        f"Calling `skew({v.tolist()})` raised an error: {e}")
            return

        if not ctx.require(f"`skew({v.tolist()})` returns a 3x3 array",
                           isinstance(got, np.ndarray) and got.shape == (3, 3),
                           f"Expected a (3, 3) numpy array, got {type(got)} with shape {getattr(got, 'shape', None)}"):
            return

        if not ctx.require("Returns the correct skew-symmetric matrix",
                           np.allclose(got, expected, atol=1e-8),
                           f"For v={v.tolist()}, expected:\n{expected}\ngot:\n{got}"):
            return

        # Check cross product equivalence: skew(v) @ w == v x w
        w = np.array([0.5, -1.0, 2.0])
        expected_cross = np.cross(v, w)
        got_cross = got @ w
        if not ctx.require("skew(v) @ w matches vector cross product np.cross(v, w)",
                           np.allclose(got_cross, expected_cross, atol=1e-8),
                           f"Matrix multiplication `skew(v) @ w` did not match `np.cross(v, w)`."):
            return


# ---------------------------------------------------------------------------
# Exercise 2 — predict_state
# ---------------------------------------------------------------------------

def check_ex2(ctx):
    """Verify `predict_state` quaternion propagation, Phi computation, and P update."""
    if not _ran(ctx):
        return

    predict_func = ctx.env.get("predict_state")
    if not ctx.require("Defines a function named `predict_state`", callable(predict_func),
                       "Define a function `predict_state(...)` to propagate quaternion, state transition matrix, and covariance."):
        return

    # Sample inputs
    estimate = qt.quaternion(1.0, 0.0, 0.0, 0.0)
    b_hat = np.array([0.01, -0.02, 0.005])
    P = np.eye(6) * 0.01
    measured_omega = np.array([0.1, 0.2, -0.15])
    dt = 0.01
    sigma_gyro = 0.02
    sigma_bias = 1e-5

    # Compute expected
    skew_func = ctx.env.get("skew")
    if skew_func is None:
        def skew_func(v):
            return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    omega_hat = measured_omega - b_hat
    expected_est = estimate + 0.5 * qt.quaternion(0, *omega_hat) * estimate * dt
    expected_est = expected_est.normalized()

    expected_Phi = np.eye(6)
    expected_Phi[0:3, 0:3] = np.eye(3) - skew_func(omega_hat) * dt
    expected_Phi[0:3, 3:6] = -np.eye(3) * dt
    expected_Phi[3:6, 3:6] = np.eye(3)

    Q_d = np.eye(6)
    Q_d[0:3, 0:3] *= dt * (sigma_gyro**2)
    Q_d[3:6, 3:6] *= dt * (sigma_bias**2)
    expected_P = expected_Phi @ P @ expected_Phi.T + Q_d

    try:
        res = predict_func(estimate, b_hat, P, measured_omega, dt, sigma_gyro, sigma_bias)
    except Exception as e:
        ctx.require("`predict_state(...)` runs without throwing an exception", False,
                    f"Calling `predict_state(...)` raised an error: {e}")
        return

    if not ctx.require("`predict_state` returns 3 outputs: (estimate, P, Phi)",
                       isinstance(res, (tuple, list)) and len(res) == 3,
                       f"Expected a tuple/list of 3 items (estimate_next, P_next, Phi), got {type(res)}"):
        return

    est_next, P_next, Phi_out = res

    # Verify quaternion estimate
    if not ctx.require("Propagates quaternion estimate correctly",
                       np.allclose([est_next.w, est_next.x, est_next.y, est_next.z],
                                   [expected_est.w, expected_est.x, expected_est.y, expected_est.z], atol=1e-6),
                       f"Quaternion propagation mismatch. Expected {expected_est}, got {est_next}"):
        return

    # Verify state transition matrix Phi
    if not ctx.require("Computes 6x6 state transition matrix Phi correctly",
                       isinstance(Phi_out, np.ndarray) and Phi_out.shape == (6, 6) and np.allclose(Phi_out, expected_Phi, atol=1e-6),
                       f"State transition matrix Phi mismatch. Expected:\n{expected_Phi}\ngot:\n{Phi_out}"):
        return

    # Verify covariance matrix P
    if not ctx.require("Propagates covariance matrix P correctly",
                       isinstance(P_next, np.ndarray) and P_next.shape == (6, 6) and np.allclose(P_next, expected_P, atol=1e-6),
                       f"Covariance P propagation mismatch. Expected:\n{expected_P}\ngot:\n{P_next}"):
        return


# ---------------------------------------------------------------------------
# Exercise 3 — update_state
# ---------------------------------------------------------------------------

def check_ex3(ctx):
    """Verify `update_state` innovation, gain, multiplicative quaternion & bias update, and covariance update."""
    if not _ran(ctx):
        return

    update_func = ctx.env.get("update_state")
    if not ctx.require("Defines a function named `update_state`", callable(update_func),
                       "Define a function `update_state(...)` for measurement update."):
        return

    # Sample inputs
    estimate = qt.quaternion(0.9998, 0.01, 0.01, 0.01).normalized()
    b_hat = np.array([0.005, -0.01, 0.002])
    P = np.eye(6) * 0.005
    measured_accel = np.array([0.15, -0.2, -9.75])
    g_ref = np.array([0.0, 0.0, -9.81])
    R = np.eye(3) * (0.03**2)

    skew_func = ctx.env.get("skew")
    if skew_func is None:
        def skew_func(v):
            return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    # Expected update step
    R_body_to_inertial = qt.as_rotation_matrix(estimate)
    a_hat = R_body_to_inertial.T @ g_ref

    H = np.zeros((3, 6))
    H[0:3, 0:3] = -skew_func(a_hat)

    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)

    r = measured_accel - a_hat
    dx = K @ r
    delta_theta = dx[0:3]
    delta_b = dx[3:6]

    delta_q = qt.quaternion(1.0, *(0.5 * delta_theta))
    expected_est = (estimate * delta_q).normalized()
    expected_b_hat = b_hat + delta_b

    I6 = np.eye(6)
    expected_P = (I6 - K @ H) @ P @ (I6 - K @ H).T + K @ R @ K.T

    try:
        res = update_func(estimate, b_hat, P, measured_accel, g_ref, R)
    except Exception as e:
        ctx.require("`update_state(...)` runs without throwing an exception", False,
                    f"Calling `update_state(...)` raised an error: {e}")
        return

    if not ctx.require("`update_state` returns 3 outputs: (estimate, b_hat, P)",
                       isinstance(res, (tuple, list)) and len(res) == 3,
                       f"Expected a tuple/list of 3 items (estimate_next, b_hat_next, P_next), got {type(res)}"):
        return

    est_next, b_hat_next, P_next = res

    # Verify updated quaternion estimate
    if not ctx.require("Updates quaternion estimate multiplicatively",
                       np.allclose([est_next.w, est_next.x, est_next.y, est_next.z],
                                   [expected_est.w, expected_est.x, expected_est.y, expected_est.z], atol=1e-5),
                       f"Updated quaternion mismatch. Expected {expected_est}, got {est_next}"):
        return

    # Verify updated bias
    if not ctx.require("Updates bias estimate correctly",
                       isinstance(b_hat_next, np.ndarray) and np.allclose(b_hat_next, expected_b_hat, atol=1e-6),
                       f"Updated bias estimate mismatch. Expected {expected_b_hat}, got {b_hat_next}"):
        return

    # Verify updated covariance
    if not ctx.require("Updates covariance matrix P correctly",
                       isinstance(P_next, np.ndarray) and P_next.shape == (6, 6) and np.allclose(P_next, expected_P, atol=1e-5),
                       f"Updated covariance P mismatch. Expected:\n{expected_P}\ngot:\n{P_next}"):
        return
