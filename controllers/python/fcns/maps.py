"""Lie algebra utility maps: hat, vee, vec.

Direct Python translation of the MATLAB functions:
    hatMap.m, veeMap.m, vec.m
"""

import numpy as np


def hatMap(a):
    """Hat map: 3-vector -> 3x3 skew-symmetric matrix.

    Matches MATLAB hatMap.m. Accepts a length-3 vector and returns a (3,3)
    array of float64.
    """
    a = np.asarray(a, dtype=float).reshape(3)
    return np.array([
        [0.0, -a[2], a[1]],
        [a[2], 0.0, -a[0]],
        [-a[1], a[0], 0.0],
    ])


def veeMap(in_):
    """Vee map: 3x3 skew-symmetric matrix -> 3-vector.

    Matches MATLAB veeMap.m.
    """
    in_ = np.asarray(in_, dtype=float).reshape(3, 3)
    return np.array([
        -in_[1, 2],
        in_[0, 2],
        -in_[0, 1],
    ])


def vec(m):
    """Column-wise vectorization, mirroring MATLAB reshape(m, [], 1)."""
    m = np.asarray(m)
    return m.reshape(-1, 1, order='F')
