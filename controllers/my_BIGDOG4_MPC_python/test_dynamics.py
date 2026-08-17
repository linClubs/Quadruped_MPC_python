"""测试李群动力学和MPC在站立姿态下的输出."""
import numpy as np
from lie_dynamics import get_AB_continuous
from solve_mpc import solve_mpc, _c2qp
from kinematics import forward_kinematics
from rotations import eta_error, euler_to_rot


def test_standing_dynamics():
    """测试站立姿态下的动力学矩阵."""
    print("=" * 60)
    print("测试: 站立姿态动力学")
    print("=" * 60)

    # 站立姿态参数
    R = np.eye(3)
    w = np.zeros(3)
    mass = 13.5
    J = np.diag([0.06150, 0.1313, 0.1646])
    x_op = np.array([0.0, 0.0, 0.3])

    # 站立关节角 (典型值: hip=0, leg=-0.45+0.45=0, foot=1.40-1.40=0)
    # 但实际站立时关节角不为0, 用近似值
    q = np.array([
        0.0, 0.5, -1.0,   # FR
        0.0, 0.5, -1.0,   # FL
        0.0, 0.5, -1.0,   # BR
        0.0, 0.5, -1.0,   # BL
    ])

    rsf, rbf = forward_kinematics(q)
    rbf_world = R @ rbf

    print(f"足端位置 (世界系, 相对躯干中心):")
    print(f"  rbf_world = \n{rbf_world}")
    print(f"  足端 z 坐标: {rbf_world[2, :]}")

    # 操作点力 (重力分配)
    fop = np.zeros(12)
    for i in range(4):
        fop[3*i + 2] = mass * 9.81 / 4.0

    print(f"\n操作点力 (重力分配): {fop}")

    # 计算连续动力学
    A_c, B_c, d_c = get_AB_continuous(R, w, fop, rbf_world, mass, J, x_op)

    print(f"\n连续动力学 A_c:")
    print(np.array2string(A_c, precision=4, suppress_small=True))
    print(f"\n连续动力学 d_c:")
    print(np.array2string(d_c, precision=4, suppress_small=True))

    print(f"\nB_c[3:6, :] (速度对力的映射):")
    print(np.array2string(B_c[3:6, :], precision=4, suppress_small=True))
    print(f"\nB_c[9:12, :] (角速度对力的映射):")
    print(np.array2string(B_c[9:12, :], precision=4, suppress_small=True))

    # 验证: d_c[3:6] 应该是 [0, 0, -9.81]
    print(f"\n验证 d_c[3:6] (重力): {d_c[3:6]} (期望 [0, 0, -9.81])")

    # 验证: d_c[9:12] (角速度常数项)
    # 站立时 w=0, 所以 -hat(w)@J@w = 0, temp_J_w@w = 0
    # Mop = sum(r_i × F_i), 如果足端对称且力均匀, Mop ≈ 0
    print(f"验证 d_c[9:12] (角速度常数项): {d_c[9:12]}")

    return A_c, B_c, d_c, rbf_world


def test_mpc_standing():
    """测试MPC在站立姿态下的力分配."""
    print("\n" + "=" * 60)
    print("测试: MPC 站立力分配")
    print("=" * 60)

    R = np.eye(3)
    w = np.zeros(3)
    mass = 13.5
    J = np.diag([0.06150, 0.1313, 0.1646])
    x_op = np.array([0.0, 0.0, 0.3])

    q = np.array([
        0.0, 0.5, -1.0,
        0.0, 0.5, -1.0,
        0.0, 0.5, -1.0,
        0.0, 0.5, -1.0,
    ])
    rsf, rbf = forward_kinematics(q)
    rbf_world = R @ rbf

    # 当前状态 [p, v, eta, w]
    x0 = np.zeros(12)
    x0[0:3] = [0.0, 0.0, 0.3]  # 位置
    x0[3:6] = [0.0, 0.0, 0.0]  # 速度
    x0[6:9] = [0.0, 0.0, 0.0]  # eta (操作点=0)
    x0[9:12] = [0.0, 0.0, 0.0]  # 角速度

    # 期望状态 [p, v, eta, w]
    horizon = 10
    dt_mpc = 0.03
    xd = np.zeros(12 * horizon)
    for i in range(horizon):
        xd[12*i + 0] = 0.0    # x
        xd[12*i + 1] = 0.0    # y
        xd[12*i + 2] = 0.3    # z (期望高度)
        xd[12*i + 3] = 0.0    # vx
        xd[12*i + 4] = 0.0    # vy
        xd[12*i + 5] = 0.0    # vz
        xd[12*i + 6] = 0.0    # eta_x
        xd[12*i + 7] = 0.0    # eta_y
        xd[12*i + 8] = 0.0    # eta_z
        xd[12*i + 9] = 0.0    # wx
        xd[12*i + 10] = 0.0   # wy
        xd[12*i + 11] = 0.0   # wz

    # 步态表 (全支撑)
    gait_table = np.ones(4 * horizon)

    # 操作点力
    fop = np.zeros(12)
    for i in range(4):
        fop[3*i + 2] = mass * 9.81 / 4.0

    f = solve_mpc(R, w, fop, rbf_world, x0, xd, dt_mpc, horizon,
                  gait_table, mass, J, x_op)

    print(f"MPC 力输出 (前12维, 第一步):")
    for i in range(4):
        print(f"  腿{i+1}: Fx={f[3*i]:.2f}, Fy={f[3*i+1]:.2f}, Fz={f[3*i+2]:.2f}")

    total_fz = sum(f[3*i+2] for i in range(4))
    print(f"\n  总 Fz = {total_fz:.2f} (期望 {mass*9.81:.2f})")

    return f


def test_mpc_height_error():
    """测试有高度误差时的MPC输出."""
    print("\n" + "=" * 60)
    print("测试: MPC 高度误差 (z=0.329 vs 期望 0.3)")
    print("=" * 60)

    R = np.eye(3)
    w = np.zeros(3)
    mass = 13.5
    J = np.diag([0.06150, 0.1313, 0.1646])

    q = np.array([
        0.0, 0.5, -1.0,
        0.0, 0.5, -1.0,
        0.0, 0.5, -1.0,
        0.0, 0.5, -1.0,
    ])
    rsf, rbf = forward_kinematics(q)
    rbf_world = R @ rbf

    # 当前状态: z=0.329 (初始高度)
    x0 = np.zeros(12)
    x0[0:3] = [0.0, 0.0, 0.329]
    x0[6:9] = [0.0, 0.0, 0.0]

    horizon = 10
    dt_mpc = 0.03
    xd = np.zeros(12 * horizon)
    for i in range(horizon):
        xd[12*i + 2] = 0.3  # 期望高度

    gait_table = np.ones(4 * horizon)

    fop = np.zeros(12)
    for i in range(4):
        fop[3*i + 2] = mass * 9.81 / 4.0

    x_op = np.array([0.0, 0.0, 0.329])

    f = solve_mpc(R, w, fop, rbf_world, x0, xd, dt_mpc, horizon,
                  gait_table, mass, J, x_op)

    print(f"MPC 力输出 (前12维, 第一步):")
    for i in range(4):
        print(f"  腿{i+1}: Fx={f[3*i]:.2f}, Fy={f[3*i+1]:.2f}, Fz={f[3*i+2]:.2f}")

    total_fz = sum(f[3*i+2] for i in range(4))
    print(f"\n  总 Fz = {total_fz:.2f} (重力 {mass*9.81:.2f})")
    print(f"  高度误差 = {0.329 - 0.3:.4f}m")

    return f


def test_mpc_attitude_error():
    """测试有姿态误差时的MPC输出."""
    print("\n" + "=" * 60)
    print("测试: MPC 姿态误差 (roll=10度)")
    print("=" * 60)

    mass = 13.5
    J = np.diag([0.06150, 0.1313, 0.1646])

    # 当前有10度roll
    roll = np.radians(10)
    R = euler_to_rot(roll, 0, 0)
    w = np.zeros(3)

    q = np.array([
        0.0, 0.5, -1.0,
        0.0, 0.5, -1.0,
        0.0, 0.5, -1.0,
        0.0, 0.5, -1.0,
    ])
    rsf, rbf = forward_kinematics(q)
    rbf_world = R @ rbf

    x_op = np.array([0.0, 0.0, 0.3])

    # 期望姿态 = 单位矩阵
    Rd = np.eye(3)
    eta_des = eta_error(R, Rd)

    print(f"当前 roll = {np.degrees(roll):.1f} 度")
    print(f"eta_des (期望姿态误差) = {eta_des}")
    print(f"eta_des (度) = {np.degrees(eta_des)}")

    x0 = np.zeros(12)
    x0[0:3] = [0.0, 0.0, 0.3]
    x0[6:9] = [0.0, 0.0, 0.0]  # 操作点 eta=0
    x0[9:12] = [0.0, 0.0, 0.0]

    horizon = 10
    dt_mpc = 0.03
    xd = np.zeros(12 * horizon)
    for i in range(horizon):
        xd[12*i + 2] = 0.3
        xd[12*i + 6:12*i + 9] = eta_des  # 期望 eta

    gait_table = np.ones(4 * horizon)

    fop = np.zeros(12)
    for i in range(4):
        fop[3*i + 2] = mass * 9.81 / 4.0

    f = solve_mpc(R, w, fop, rbf_world, x0, xd, dt_mpc, horizon,
                  gait_table, mass, J, x_op)

    print(f"\nMPC 力输出 (前12维, 第一步):")
    for i in range(4):
        print(f"  腿{i+1}: Fx={f[3*i]:.2f}, Fy={f[3*i+1]:.2f}, Fz={f[3*i+2]:.2f}")

    total_fz = sum(f[3*i+2] for i in range(4))
    print(f"\n  总 Fz = {total_fz:.2f} (重力 {mass*9.81:.2f})")

    # 计算恢复力矩
    tau = np.zeros(3)
    for i in range(4):
        tau += np.cross(rbf_world[:, i], f[3*i:3*i+3])
    print(f"  恢复力矩 = {tau}")
    print(f"  期望: x方向力矩应为负 (纠正roll)")

    return f


def compare_with_matlab_dynamics():
    """对比MATLAB简化动力学与李群动力学."""
    print("\n" + "=" * 60)
    print("对比: MATLAB简化动力学 vs 李群动力学")
    print("=" * 60)

    mass = 13.5
    J = np.diag([0.06150, 0.1313, 0.1646])
    R = np.eye(3)
    w = np.zeros(3)
    g = 9.81

    q = np.array([
        0.0, 0.5, -1.0,
        0.0, 0.5, -1.0,
        0.0, 0.5, -1.0,
        0.0, 0.5, -1.0,
    ])
    rsf, rbf = forward_kinematics(q)
    rbf_world = R @ rbf

    # ===== MATLAB 13D 简化动力学 =====
    # 状态: [rpy(3), p(3), w(3), v(3), g(1)]
    Ac_matlab = np.zeros((13, 13))
    Ac_matlab[0:3, 6:9] = R.T   # rpy_dot = R' @ w (简化: 只用yaw)
    Ac_matlab[3:6, 9:12] = np.eye(3)  # p_dot = v
    Ac_matlab[11, 12] = 1.0     # v_z_dot += g

    I_world = R @ J @ R.T
    I_inv = np.linalg.inv(I_world)
    Bc_matlab = np.zeros((13, 12))
    for i in range(4):
        ri = rbf_world[:, i]
        Bc_matlab[6:9, 3*i:3*i+3] = I_inv @ np.array([
            [0, -ri[2], ri[1]],
            [ri[2], 0, -ri[0]],
            [-ri[1], ri[0], 0]
        ])
        Bc_matlab[9:12, 3*i:3*i+3] = np.eye(3) / mass

    print("MATLAB Ac (非零元素):")
    for i in range(13):
        for j in range(13):
            if abs(Ac_matlab[i, j]) > 1e-10:
                print(f"  Ac[{i},{j}] = {Ac_matlab[i,j]:.4f}")

    print("\nMATLAB Bc (非零元素):")
    for i in range(13):
        for j in range(12):
            if abs(Bc_matlab[i, j]) > 1e-10:
                print(f"  Bc[{i},{j}] = {Bc_matlab[i,j]:.4f}")

    # ===== 李群 12D 动力学 =====
    x_op = np.array([0.0, 0.0, 0.3])
    fop = np.zeros(12)
    for i in range(4):
        fop[3*i + 2] = mass * g / 4.0

    A_c, B_c, d_c = get_AB_continuous(R, w, fop, rbf_world, mass, J, x_op)

    print("\n李群 A_c (非零元素):")
    for i in range(12):
        for j in range(12):
            if abs(A_c[i, j]) > 1e-10:
                print(f"  A_c[{i},{j}] = {A_c[i,j]:.4f}")

    print("\n李群 B_c (非零元素):")
    for i in range(12):
        for j in range(12):
            if abs(B_c[i, j]) > 1e-10:
                print(f"  B_c[{i},{j}] = {B_c[i,j]:.4f}")

    print("\n李群 d_c:")
    for i in range(12):
        if abs(d_c[i]) > 1e-10:
            print(f"  d_c[{i}] = {d_c[i]:.4f}")

    # 关键对比: 速度对力的映射
    print("\n=== 关键对比 ===")
    print(f"MATLAB Bc[9:12, 3*i:3*i+3] (v_dot/F): {1.0/mass:.6f}")
    print(f"李群   B_c[3:6, 3*i:3*i+3]  (v_dot/F): {B_c[3,0]:.6f}")

    # 角速度对力的映射
    print(f"\nMATLAB Bc[6:9, 0:3] (w_dot/F, 腿1):")
    print(np.array2string(Bc_matlab[6:9, 0:3], precision=6))
    print(f"李群   B_c[9:12, 0:3] (w_dot/F, 腿1):")
    print(np.array2string(B_c[9:12, 0:3], precision=6))


if __name__ == "__main__":
    test_standing_dynamics()
    test_mpc_standing()
    test_mpc_height_error()
    test_mpc_attitude_error()
    compare_with_matlab_dynamics()
