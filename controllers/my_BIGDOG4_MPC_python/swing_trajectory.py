import numpy as np


def _cubic_bezier(p0, pf, t):
    return p0 + (t**3 + 3 * t**2 * (1 - t)) * (pf - p0)


def _cubic_bezier_v(p0, pf, t):
    return 6 * t * (1 - t) * (pf - p0)


def _cubic_bezier_a(p0, pf, t):
    return (6 - 12 * t) * (pf - p0)


def swing_trajectory_bezier(pf_init, pf_final, phase, swingtime, h):
    """
    摆动腿贝塞尔轨迹 (对应 MATLAB SwingTrajectoryBezier.m)
    返回: 位置, 速度, 加速度 (各为 3 维向量)
    """
    pout = _cubic_bezier(pf_init, pf_final, phase)
    p_v = _cubic_bezier_v(pf_init, pf_final, phase) / swingtime
    p_a = _cubic_bezier_a(pf_init, pf_final, phase) / (swingtime * swingtime)

    if phase < 0.5:
        zp = _cubic_bezier(pf_init[2], pf_init[2] + h, phase * 2)
        zv = _cubic_bezier_v(pf_init[2], pf_init[2] + h, phase * 2) * 2 / swingtime
        za = _cubic_bezier_a(pf_init[2], pf_init[2] + h, phase * 2) * 4 / (swingtime * swingtime)
    else:
        zp = _cubic_bezier(pf_init[2] + h, pf_final[2], phase * 2 - 1)
        zv = _cubic_bezier_v(pf_init[2] + h, pf_final[2], phase * 2 - 1) * 2 / swingtime
        za = _cubic_bezier_a(pf_init[2] + h, pf_final[2], phase * 2 - 1) * 4 / (swingtime * swingtime)

    pout[2] = zp
    p_v[2] = zv
    p_a[2] = za
    return pout, p_v, p_a
