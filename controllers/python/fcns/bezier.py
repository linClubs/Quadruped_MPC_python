"""Bezier polynomial utilities.

Python translation of bz_int.m and polyval_bz.m.
"""

import math
import numpy as np


def polyval_bz(alpha, s):
    """Evaluate a Bezier polynomial.

    alpha : (M+1,) or (M+1, K) array of Bezier coefficients.
    s     : scalar or array of parameter values in [0, 1].

    Returns the same shape as ``s`` (with trailing K dimension when alpha is
    2-D), matching the broadcasting behaviour of the MATLAB version.
    """
    alpha = np.asarray(alpha, dtype=float)
    s = np.asarray(s, dtype=float)

    # Ensure alpha is 2-D with shape (M+1, K)
    if alpha.ndim == 1:
        alpha2 = alpha.reshape(-1, 1)
    else:
        alpha2 = alpha
    M = alpha2.shape[0] - 1

    # Expand s to broadcast with alpha along the coefficient axis.
    # Final shape: (M+1, *s.shape)
    alpha_b = alpha2.reshape((M + 1,) + (1,) * s.ndim)
    s_b = s.reshape((1,) + s.shape)
    k = np.arange(M + 1).reshape((M + 1,) + (1,) * s.ndim)
    coef = np.array([math.factorial(M) / (math.factorial(int(kk)) * math.factorial(M - int(kk)))
                     for kk in range(M + 1)]).reshape((M + 1,) + (1,) * s.ndim)
    b = (alpha_b * coef * (s_b ** k) * ((1.0 - s_b) ** (M - k))).sum(axis=0)
    return b


def bz_int(alpha, x0, s_max=1.0):
    """Integrate a Bezier polynomial so its values lie on a Bezier curve of one
    higher order, fixing the integral's value at ``s = 0`` to ``x0``.

    Mirrors MATLAB bz_int.m. ``alpha`` is a 1-D array of coefficients; returns
    a 1-D array of length ``len(alpha) + 1``.
    """
    alpha = np.asarray(alpha, dtype=float).reshape(-1)
    M = alpha.size

    # Build the integration matrix AA of shape (M+1, M+1).
    AA = np.zeros((M + 1, M + 1))
    for ii in range(M):
        AA[ii, ii] = -1.0
        AA[ii, ii + 1] = 1.0
    AA = (M / s_max) * AA
    AA[M, 0] = 1.0

    rhs = np.concatenate([alpha, [x0]])
    # Solve AA \ rhs ; MATLAB solves column vectors, here we use 1-D.
    alpha_int = np.linalg.solve(AA, rhs)
    return alpha_int
