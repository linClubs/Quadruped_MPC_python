"""李群 SO(3) 工具: hat/vee 映射 + 旋量姿态误差."""
import numpy as np
from scipy.linalg import logm


def hat_map(a):
    """3 向量 -> 3x3 反对称矩阵 (hat map)."""
    a = np.asarray(a, dtype=float).reshape(3)
    return np.array([
        [0.0, -a[2], a[1]],
        [a[2], 0.0, -a[0]],
        [-a[1], a[0], 0.0],
    ])


def vee_map(m):
    """3x3 反对称矩阵 -> 3 向量 (vee map)."""
    m = np.asarray(m, dtype=float).reshape(3, 3)
    return np.array([-m[1, 2], m[0, 2], -m[0, 1]])


def euler_to_rot(roll, pitch, yaw):
    """欧拉角 (roll,pitch,yaw) -> 旋转矩阵 R = Rz(yaw)·Ry(pitch)·Rx(roll)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def eta_error(Rd, R):
    """旋量姿态误差 η = veeMap(logm(Rdᵀ·R)).

    用于替代欧拉角差, 避免大角度奇异.
    Rd: 期望旋转矩阵, R: 当前旋转矩阵.
    """
    return vee_map(logm(Rd.T @ R))
