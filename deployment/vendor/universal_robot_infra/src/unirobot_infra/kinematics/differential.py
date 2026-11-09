"""Wrist-first differential IK with an optional elbow null-space task."""

import numpy as np
from scipy.optimize import lsq_linear


def wrist_and_elbow_steps(jacobian, error, elbow_jacobian, elbow_error,
                          joints, home, limits, elbow_gain):
    """Return a bounded wrist step and a secondary step that preserves it.

    The primary least-squares problem contains only wrist pose error. The
    elbow and home preferences act in the *exact* SVD null space, not the
    approximate null space of a damped inverse. Joints at a bound have no
    remaining freedom for the secondary task. Finite-step FK is checked by
    the caller before accepting the elbow correction.
    """
    lower = np.maximum(limits[:, 0] - joints, -.08)
    upper = np.minimum(limits[:, 1] - joints, .08)
    damping = .01
    system = np.vstack((jacobian, damping * np.eye(len(joints))))
    rhs = np.r_[error, np.zeros(len(joints))]
    primary = np.linalg.lstsq(system, rhs, rcond=None)[0]
    if np.any(primary < lower) or np.any(primary > upper):
        primary = lsq_linear(system, rhs, bounds=(lower, upper),
                             method="bvls", tol=1e-9).x

    secondary = np.zeros_like(primary)
    if elbow_gain == 0:
        return primary, secondary
    free = (primary > lower + 1e-7) & (primary < upper - 1e-7)
    if not np.any(free):
        return primary, secondary
    _, singular, vt = np.linalg.svd(jacobian[:, free], full_matrices=True)
    rank = np.count_nonzero(singular > 1e-6)
    basis = np.zeros((len(joints), vt.shape[0] - rank))
    basis[free] = vt[rank:].T
    if basis.shape[1] == 0:
        return primary, secondary

    projected = elbow_jacobian @ basis
    residual = elbow_error - elbow_jacobian @ primary
    # Mild home preference only in the remaining redundancy. It cannot
    # drag the wrist back to home as the former full-joint penalty did.
    coeff = np.linalg.solve(projected.T @ projected + .03**2 * np.eye(basis.shape[1]),
                            projected.T @ residual
                            + .0002 * basis.T @ (home - joints))
    secondary = elbow_gain * basis @ coeff
    # Uniform scaling preserves J_wrist @ secondary == 0. Per-joint clipping
    # would destroy that property and reintroduce wrist/elbow competition.
    fraction = min(1., .04 / max(np.max(np.abs(secondary)), 1e-12))
    positive = secondary > 1e-12
    negative = secondary < -1e-12
    if np.any(positive):
        fraction = min(fraction, np.min((upper - primary)[positive] / secondary[positive]))
    if np.any(negative):
        fraction = min(fraction, np.min((lower - primary)[negative] / secondary[negative]))
    return primary, secondary * max(0., fraction)
