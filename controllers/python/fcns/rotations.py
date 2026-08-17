"""Elementary rotation matrices.

Python translation of rx.m, ry.m, rz.m.
"""

import numpy as np


def rx(phi):
    """Rotation about x-axis by angle phi (radians)."""
    c = np.cos(phi)
    s = np.sin(phi)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ])


def ry(theta):
    """Rotation about y-axis by angle theta (radians)."""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ])


def rz(psi):
    """Rotation about z-axis by angle psi (radians)."""
    c = np.cos(psi)
    s = np.sin(psi)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])
