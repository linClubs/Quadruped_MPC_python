"""Inverse kinematics and rotation-to-Euler conversion.

Python translation of fcn_invKin3.m and fcn_X2EA.m.
"""

import numpy as np
from scipy.linalg import logm

from .maps import veeMap
from .rotations import rx, ry


def fcn_X2EA(X):
    """Convert rotation matrix inside state to a (3,) Euler-like vector.

    Mirrors MATLAB fcn_X2EA.m: returns veeMap(logm(R)) as a 1-D row vector.
    """
    vR = X[6:15]
    R = vR.reshape(3, 3, order='F')
    return veeMap(logm(R))


def fcn_invKin3(X, pf, p):
    """3-DOF leg inverse kinematics.

    X  : full state vector (uses pc and R).
    pf : foot position in world frame (length-3 vector or 3xN matrix).
    p  : parameter dict (must contain L, W, d, sign_L, sign_d, l1, l2).
    """
    pf = np.asarray(pf, dtype=float).reshape(-1, 1)
    pcom = X[0:3].reshape(3, 1)
    R = X[6:15].reshape(3, 3, order='F')

    L = p['L']
    W = p['W']
    sign_L = p['sign_L']
    sign_d = p['sign_d']

    Twd2com = np.block([
        [R, pcom],
        [np.zeros((1, 3)), np.array([[1.0]])],
    ])
    Tcom2h = np.block([
        [np.eye(3), np.array([[sign_L * L / 2.0, sign_d * W / 2.0, 0.0]]).T],
        [np.zeros((1, 3)), np.array([[1.0]])],
    ])
    Twd2h = Twd2com @ Tcom2h
    p_f_wd = pf[0:3].reshape(3, 1)
    p_h_wd = Twd2h[0:3, 3].reshape(3, 1)

    p_h2f_wd = p_f_wd - p_h_wd
    p_h2f_b = R.T @ p_h2f_wd
    return _invKin(p_h2f_b, p)


def _invKin(p_h2f_b, p):
    l1 = p['l1']
    l2 = p['l2']
    d = p['d']
    sign_d = p['sign_d']

    vp = np.asarray(p_h2f_b, dtype=float).reshape(3)

    # q1
    vpyz = vp[1:3]
    ryz = float(np.linalg.norm(vpyz))
    a = float(np.arcsin(vp[1] / ryz))
    b = float(np.arcsin(d / ryz))
    q1 = a - sign_d * b

    # q2
    r = float(np.linalg.norm(vp))
    vd = sign_d * np.array([0.0, d * np.cos(q1), d * np.sin(q1)])
    rf = float(np.linalg.norm(vp - vd))
    vf = rx(q1).T @ (vp - vd)
    if vf[2] <= 0:
        a = float(np.arccos(vf[0] / rf))
    else:
        if vf[0] >= 0:
            a = float(-np.arcsin(vf[2] / rf))
        else:
            a = float(np.pi + np.arcsin(vf[2] / rf))
    cosb = (l1 ** 2 + rf ** 2 - l2 ** 2) / (2.0 * l1 * rf)
    b = float(np.arccos(cosb))
    q2 = a + b

    # q3
    cosc = (l1 ** 2 + l2 ** 2 - rf ** 2) / (2.0 * l1 * l2)
    c = float(np.arccos(cosc))
    q3 = -(np.pi - c)

    return np.array([q1, q2, q3])
