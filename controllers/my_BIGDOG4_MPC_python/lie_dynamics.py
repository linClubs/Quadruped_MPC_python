"""连续时间 12 维李群单刚体动力学.

状态 x = [p(3), v(3), η(3), ω(3)] = 12
输入 u = [F1, F2, F3, F4] (12, 世界系足端力)

η = veeMap(logm(Rdᵀ·R)) 旋量姿态误差, 操作点 R_op = R (当前).
操作点处 η = 0, 沿操作点 Taylor 线性化.

参考: controllers/python/fcns_mpc/mpc.py (Representation-Free MPC).
"""
import numpy as np

from rotations import hat_map


def _get_N():
    """9x3 投影矩阵 N (对应 python/ _fcn_get_N)."""
    return np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, -1, 0],
        [0, 0, -1],
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 0],
    ], dtype=float)


def _get_D(w):
    """9x3 矩阵 D(w) (对应 python/ _fcn_get_D)."""
    d, e, f = w.reshape(3)
    return np.array([
        [0, 0, 0],
        [e, -d, 0],
        [f, 0, -d],
        [-e, d, 0],
        [0, 0, 0],
        [0, f, -e],
        [-f, 0, d],
        [0, -f, e],
        [0, 0, 0],
    ], dtype=float)


def _get_F(k):
    """3x9 对角投影 F(k) (对应 python/ _fcn_get_F)."""
    k = k.reshape(3)
    return np.array([
        [k[0], 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, k[1], 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, k[2], 0, 0],
    ], dtype=float)


def get_AB_continuous(R, w, fop, rbf_world, mass, J, x_op):
    """构造连续时间 12 维李群动力学 A_c, B_c, d_c.

    参数:
      R: 3x3 当前旋转矩阵 (操作点)
      w: 3 体角速度 (操作点)
      fop: 12 操作点力 (世界系), None 则用重力分配
      rbf_world: 3x4 足端相对机体位置 (世界系, ≈ 足端-质心)
      mass: 质量
      J: 3x3 体转动惯量
      x_op: 3 当前 COM 世界位置 (用于角速度常数项)

    返回: A_c (12x12), B_c (12x12), d_c (12,)
    """
    N = _get_N()
    invN = np.linalg.pinv(N)
    I3 = np.eye(3)
    g = 9.81

    if fop is None:
        fop = np.zeros(12)
        for i in range(4):
            fop[3*i + 2] = mass * g / 4.0

    fop = fop.reshape(12)
    w = w.reshape(3)
    x_op = np.asarray(x_op, dtype=float).reshape(3)

    A_c = np.zeros((12, 12))
    B_c = np.zeros((12, 12))
    d_c = np.zeros(12)

    # 1. 位置: ṗ = v
    A_c[0:3, 3:6] = I3

    # 2. 线速度: v̇ = (1/m)ΣF - g·e_z
    for i in range(4):
        B_c[3:6, 3*i:3*i+3] = I3 / mass
    d_c[3:6] = np.array([0.0, 0.0, -g])

    # 3. η 演化 (连续, 从 python/ _eta_co_R 反推)
    #    A_c[η,η] = invN @ kron(I, hat(w)) @ N + invN @ D(w)
    #    A_c[η,ω] = I
    #    d_c[η] = 0 (推导: invN@vec(hat(w)) - w = 0)
    A_c[6:9, 6:9] = (invN @ np.kron(I3, hat_map(w)) @ N
                     + invN @ _get_D(w))
    A_c[6:9, 9:12] = I3

    # 4. 角速度演化 (连续, 从 python/ _eta_co_w 反推)
    sum_fop = (fop[0:3] + fop[3:6] + fop[6:9] + fop[9:12])
    r1, r2, r3, r4 = rbf_world[:, 0], rbf_world[:, 1], rbf_world[:, 2], rbf_world[:, 3]
    Mop = np.hstack([hat_map(r1), hat_map(r2), hat_map(r3), hat_map(r4)]) @ fop

    temp_J_w = hat_map(J @ w) - hat_map(w) @ J  # 陀螺力矩线性化项

    Cx = R.T @ hat_map(sum_fop)                                  # ω 对 p 的依赖
    Ceta = _get_F(R.T @ Mop) @ N - temp_J_w @ hat_map(w)         # ω 对 η 的依赖
    Cw = temp_J_w                                                # ω 对 ω 的依赖
    Cu = R.T @ np.hstack([hat_map(r1), hat_map(r2), hat_map(r3), hat_map(r4)])

    A_c[9:12, 0:3] = np.linalg.solve(J, Cx)
    A_c[9:12, 6:9] = np.linalg.solve(J, Ceta)
    A_c[9:12, 9:12] = np.linalg.solve(J, Cw)
    B_c[9:12, :] = np.linalg.solve(J, Cu)

    # 角速度常数项
    # 线性化在 (x_op, w_op, fop) 处展开, 输入为绝对力 u (非偏差 u-fop).
    # 推导: ω̇ = J⁻¹(-hat(w)@J@w + Rᵀ@Σ(rᵢ×Fᵢ))
    #   在操作点 Taylor 展开: ω̇ ≈ f(x_op,w_op,u) + A_ωx·δx + A_ωw·δw + B_ωu·δu
    #   其中 δu = u - fop, B_ωu = J⁻¹@Cu
    #   展开后常数项含 Rᵀ@Mop (来自 f(x_op,u_op)), 而输入项含 Cu@(u-fop)
    #   = Cu@u - Cu@fop = Cu@u - Rᵀ@Mop  → Rᵀ@Mop 恰好抵消.
    # 因此 Cc = -hat(w)@J@w - temp_J_w@w - Cx@x_op (不含 Rᵀ@Mop)
    Cc = (-hat_map(w) @ J @ w
          - temp_J_w @ w
          - Cx @ x_op)
    d_c[9:12] = np.linalg.solve(J, Cc)

    return A_c, B_c, d_c
