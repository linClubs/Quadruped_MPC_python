import numpy as np
from scipy.linalg import expm

from lie_dynamics import get_AB_continuous
from qp_wrapper import solve_qp


def _c2qp(A, B, d, dt, horizon):
    """连续到离散的 QP 展开矩阵 (ZOH 精确离散化).

    12 维状态, 12 维输入, 含常数项 d (重力).
    返回 A_qp, B_qp, d_qp 使得 x_pred = A_qp @ x0 + B_qp @ u + d_qp.
    """
    n = 12
    # 增广矩阵 [[A, B, d], [0, 0, 0], [0, 0, 0]] (25x25)
    ABc = np.zeros((2 * n + 1, 2 * n + 1))
    ABc[0:n, 0:n] = A
    ABc[0:n, n:2 * n] = B
    ABc[0:n, 2 * n] = d
    ABc = dt * ABc
    expmm = expm(ABc)

    Adt = expmm[0:n, 0:n]
    Bdt = expmm[0:n, n:2 * n]
    ddt = expmm[0:n, 2 * n]  # 离散常数项

    powerMats = np.zeros((n, n, horizon + 1))
    powerMats[:, :, 0] = np.eye(n)
    for i in range(1, horizon + 1):
        powerMats[:, :, i] = Adt @ powerMats[:, :, i - 1]

    A_qp = np.zeros((n * horizon, n))
    B_qp = np.zeros((n * horizon, n * horizon))
    d_qp = np.zeros(n * horizon)
    cum_d = np.zeros(n)
    for m in range(horizon):
        A_qp[n*m:n*m+n, 0:n] = powerMats[:, :, m + 1]
        for n_idx in range(horizon):
            if m >= n_idx:
                a_num = m - n_idx
                B_qp[n*m:n*m+n, n*n_idx:n*n_idx+n] = powerMats[:, :, a_num] @ Bdt
        cum_d = Adt @ cum_d + ddt
        d_qp[n*m:n*m+n] = cum_d

    return A_qp, B_qp, d_qp


def solve_mpc(R, w, fop, rbf_world, x0, xd, dt, horizon, gait_table, mass, J, x_op):
    """
    李群 MPC 求解 (12 维状态 [p, v, η, ω])

    输入:
      R: 3x3 当前旋转矩阵 (操作点)
      w: 3 体角速度
      fop: 12 操作点力 (上一次解或 None)
      rbf_world: 3x4 足端位置 (世界系)
      x0: 12 当前状态 [p, v, η, ω] (操作点处 η=0)
      xd: 12*horizon 期望状态
      dt: MPC 离散时间步长
      horizon: 预测步数
      gait_table: 4*horizon 步态表 (1=支撑, 0=摆动)
      mass: 质量
      J: 3x3 体转动惯量
      x_op: 3 当前 COM 世界位置 (用于角速度常数项)
    返回: 12*horizon 最优力序列
    """
    f_max = 140
    mu = 1.0 / 0.4
    alpha = 0.00002

    # 连续时间李群动力学 A_c, B_c, d_c (12x12, 12x12, 12)
    A_c, B_c, d_c = get_AB_continuous(R, w, fop, rbf_world, mass, J, x_op)

    # 摩擦锥约束矩阵 (20*horizon x 12*horizon)
    f_block = np.array([
        [mu, 0, 1],
        [-mu, 0, 1],
        [0, mu, 1],
        [0, -mu, 1],
        [0, 0, 1],
    ])
    A_constr = np.zeros((20 * horizon, 12 * horizon))
    for i in range(4 * horizon):
        A_constr[5*i:5*i+5, 3*i:3*i+3] = f_block

    # ZOH 精确离散化 (含重力常数项)
    A_qp, B_qp, d_qp = _c2qp(A_c, B_c, d_c, dt, horizon)

    # 权重矩阵 S (12 维状态: [p, v, η, ω])
    # MATLAB 13维状态 [rpy, p, w, v, g] 权重 [25,25,10, 2,2,100, 0,0,0.3, 10,10,20, 0]
    # 映射到李群12维 [p, v, η, ω]:
    #   p  ← MATLAB p     = [2, 2, 100]
    #   v  ← MATLAB v     = [10, 10, 20]
    #   η  ← MATLAB rpy   = [25, 25, 10]
    #   ω  ← MATLAB w     = [0, 0, 0.3]
    full_weight = np.array([2, 2, 100,         # 位置 [x, y, z]
                            10, 10, 20,        # 线速度 [vx, vy, vz]
                            25, 25, 0,         # η 姿态误差 (yaw 权重=0, 由 w_ref 跟踪)
                            0, 0, 0.3])        # 角速度 [wx, wy, wz]
    n = 12
    S = np.zeros((n * horizon, n * horizon))
    for i in range(horizon):
        for j in range(n):
            S[i*n + j, i*n + j] = full_weight[j]

    # QP 目标: min 0.5*x'Hx + g'x
    # 预测: x_pred = A_qp @ x0 + B_qp @ u + d_qp
    H = 2 * (B_qp.T @ S @ B_qp + alpha * np.eye(12 * horizon))
    g = 2 * B_qp.T @ S @ (A_qp @ x0 + d_qp - xd)

    # 约束边界
    lbA = np.zeros(20 * horizon)
    ubA = np.zeros(20 * horizon)
    k = 0
    for i in range(horizon):
        for j in range(4):
            lbA[5*k:5*k+5] = 0
            ubA[5*k + 0] = 100000
            ubA[5*k + 1] = 100000
            ubA[5*k + 2] = 100000
            ubA[5*k + 3] = 100000
            ubA[5*k + 4] = gait_table[4*i + j] * f_max
            k += 1

    x = solve_qp(H, g, A=A_constr, lb=None, ub=None, lbA=lbA, ubA=ubA)
    return x
