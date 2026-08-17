import numpy as np

from kinematics import skew
from qp_wrapper import solve_qp


def qp_standing(rbf, b_control, flag):
    """
    站立时的力分配 QP 求解 (对应 MATLAB qp.m)

    输入:
      rbf: 3x4 足端位置(世界系)
      b_control: 6x1 [力; 力矩]
      flag: 4x1 接触标志 (1=接触, 0=不接触)
    返回: 12x1 地面反力
    """
    uf = 0.5
    n = 12

    ub = np.zeros(n)
    lb = np.zeros(n)

    # 初始约束边界 (20个约束: 每条腿5个)
    ubA = np.array([0, 1e6, 0, 1e6, 160,
                    0, 1e6, 0, 1e6, 160,
                    0, 1e6, 0, 1e6, 160,
                    0, 1e6, 0, 1e6, 160], dtype=np.float64)
    lbA = np.array([0, 1e6, 0, 1e6, 10,
                    0, 1e6, 0, 1e6, 10,
                    0, 1e6, 0, 1e6, 10,
                    0, 1e6, 0, 1e6, 10], dtype=np.float64)

    # 构造 A_control (6x12): [flag*I; flag*Skew(rbf)]
    A_control = np.zeros((6, 12))
    for i in range(4):
        A_control[0:3, 3*i:3*i+3] = flag[i] * np.eye(3)
        A_control[3:6, 3*i:3*i+3] = flag[i] * skew(rbf[:, i])

    S = np.diag([1, 1, 10, 50, 30, 10])
    W = 0.001 * np.eye(12)
    alpha = 0.01

    H = 2 * A_control.T @ S @ A_control + 2 * alpha * W
    g = -2 * A_control.T @ S @ b_control

    # 摩擦锥约束矩阵
    c = np.array([
        [1, 0, -uf],
        [1, 0, uf],
        [0, 1, -uf],
        [0, 1, uf],
        [0, 0, 1],
    ])
    A = np.zeros((20, 12))
    for i in range(4):
        A[5*i:5*i+5, 3*i:3*i+3] = flag[i] * c

    # 更新边界
    for i in range(4):
        for j in range(3):
            ub[3*i + j] = flag[i] * 100000
            lb[3*i + j] = -flag[i] * 100000

        lbA[5*i + 0] = -flag[i] * 100000
        lbA[5*i + 1] = 0
        lbA[5*i + 2] = -flag[i] * 100000
        lbA[5*i + 3] = 0
        lbA[5*i + 4] = flag[i] * 10

        ubA[5*i + 0] = 0
        ubA[5*i + 1] = flag[i] * 100000
        ubA[5*i + 2] = 0
        ubA[5*i + 3] = flag[i] * 100000
        ubA[5*i + 4] = flag[i] * 160

    x = solve_qp(H, g, A=A, lb=lb, ub=ub, lbA=lbA, ubA=ubA)
    return x
