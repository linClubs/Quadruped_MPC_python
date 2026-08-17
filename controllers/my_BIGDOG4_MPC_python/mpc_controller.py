import numpy as np

from kinematics import forward_kinematics, jacobian_matrix, skew
from rotations import euler_to_rot
from gait import set_iterations, get_swing_state, get_mpc_table
from swing_trajectory import swing_trajectory_bezier
from qp_solver import qp_standing
from solve_mpc import solve_mpc


class MPCController:
    """
    MPC 控制器 (对应 MATLAB mpcController.m)
    将 MATLAB persistent 变量转为类实例属性
    """

    def __init__(self, rbs, mass, I, offset, Kpcom, Kdcom, Kpbase, Kdbase,
                 dt, iterations_between_mpc, stancetime, swingtime, height,
                 horizon, Kp_cartesian, Kd_cartesian, vz, n_iterations):
        # 常量参数
        self.rbs = np.array(rbs, dtype=np.float64)
        self.mass = mass
        self.I = np.array(I, dtype=np.float64)
        self.offset = np.array(offset, dtype=np.float64)
        self.Kpcom = np.array(Kpcom, dtype=np.float64)
        self.Kdcom = np.array(Kdcom, dtype=np.float64)
        self.Kpbase = np.array(Kpbase, dtype=np.float64)
        self.Kdbase = np.array(Kdbase, dtype=np.float64)
        self.dt = dt
        self.iterations_between_mpc = iterations_between_mpc
        self.stancetime = stancetime
        self.swingtime = swingtime
        self.height = height
        self.horizon = horizon
        self.Kp_cartesian = np.array(Kp_cartesian, dtype=np.float64)
        self.Kd_cartesian = np.array(Kd_cartesian, dtype=np.float64)
        self.vz = vz
        self.n_iterations = n_iterations

        self.dt_mpc = dt * iterations_between_mpc

        # persistent 变量初始化
        self.state = 0
        self.timer = 0
        self.iterationcounter = 0
        self.firstswing = np.array([0, 0, 0, 0], dtype=np.float64)
        self.x_s = np.zeros(12)
        self.x_desire = np.zeros(12 * horizon)
        self.f = np.zeros(12 * horizon)
        self.swingTimeRemaining = np.zeros(4)
        self.pf_init = np.zeros((3, 4))
        self.pf_final = np.zeros((3, 4))
        self.wf = np.zeros(3)
        self.x0 = 0.0
        self.y0 = 0.0
        self.z0 = 0.0
        self.v0 = 0.0
        self._position_captured = False

    def compute(self, R, w, x, y, z, v, q1, q2, q3, q4,
                w1, w2, w3, w4, t, foot_senser, vx, vy, v_yaw,
                offsets, durations, bodyheight,
                d_roll, d_pitch, d_yaw):
        """
        计算 4 条腿的关节力矩

        参数 foot_senser: 足端接触数据, 支持两种格式
          - 1D (4,): 触摸传感器值 (1=接触 0=离地)
          - 2D (3,4): 关节力矩反馈, 内部通过雅可比反推接触力
        参数 v_yaw: 偏航角速度参考 (rad/s), 用于 MPC 角速度跟踪和落足点 yaw 补偿
        返回: tao1, tao2, tao3, tao4 (各 3x1)
        """
        trajAll = np.zeros(12 * self.horizon)
        tao = np.zeros((3, 4))
        p = np.zeros((3, 4))
        pv = np.zeros((3, 4))
        pa = np.zeros((3, 4))
        rsf_des = np.zeros((3, 4))
        vleg_des = np.zeros((3, 4))
        v_leg = np.zeros((3, 4))

        offsets = np.array(offsets, dtype=np.float64).flatten()
        durations = np.array(durations, dtype=np.float64).flatten()

        w_leg = np.column_stack([w1, w2, w3, w4])

        offsets_float = offsets / self.n_iterations
        durations_float = durations / self.n_iterations

        # 期望旋转矩阵 (李群: 用 Rd 替代欧拉角, 避免奇异)
        Rd = euler_to_rot(d_roll, d_pitch, d_yaw)
        I_world = R @ self.I @ R.T

        # 期望速度和位置
        if 3 < t < 5:
            v_ref = np.array([0, 0, self.vz], dtype=np.float64)
            r_ref = np.array([0, 0, 0.1 + self.vz * (t - 3)], dtype=np.float64)
        else:
            if t >= 5 and not self._position_captured:
                self.x0 = x
                self.y0 = y
                self.z0 = z
                self._position_captured = True
            # vx/vy 为 body 系输入, 转换到世界系 (v, r 均为世界系)
            v_ref = R @ np.array([vx, vy, 0], dtype=np.float64)
            r_ref = np.array([self.x0, self.y0, bodyheight], dtype=np.float64)

        # w_ref 为 body 系角速度 (与 gyro 的 w 同系), yaw 分量经 R 后 z 不变
        w_ref = np.array([0, 0, v_yaw], dtype=np.float64)

        r = np.array([x, y, z], dtype=np.float64)
        a = self.Kpcom @ (r_ref - r) + self.Kdcom @ (v_ref - v)
        wd = np.array([0, 0, v_yaw], dtype=np.float64)
        # 姿态误差: 直接从 R 第三行提取 roll/pitch, 避免 yaw 跨越 ±π 时
        # 旋转向量 (logm) 不连续导致 roll/pitch 反馈方向反转
        # R[2,0], R[2,1], R[2,2] 只依赖 roll/pitch, 不受 yaw 影响
        roll_cur  = np.arctan2(R[2, 1], R[2, 2])
        pitch_cur = np.arcsin(-np.clip(R[2, 0], -1.0, 1.0))
        qw = np.array([d_roll - roll_cur, d_pitch - pitch_cur, 0.0])
        aw = self.Kpbase @ qw + self.Kdbase @ (wd - w)
        F = self.mass * (a + np.array([0, 0, 9.81]))
        Tao = I_world @ aw
        b_control = np.concatenate([F, Tao])

        q = np.concatenate([q1, q2, q3, q4])

        rsf_body, rbf_body = forward_kinematics(q)
        rbf_world = R @ rbf_body
        J = jacobian_matrix(q)

        # 足端接触状态检测 (支持两种方式)
        foot_senser_arr = np.asarray(foot_senser)
        if foot_senser_arr.ndim == 1 and foot_senser_arr.size == 4:
            # 方式1: 触摸传感器 (1=接触 0=离地, 对应 MATLAB foot_senser)
            foot_contact = (foot_senser_arr > 0.5).astype(float)
        else:
            # 方式2: 关节力矩反推
            # tau = J^T @ F_body  =>  F_body = (J^T)^-1 @ tau
            # F_world = R @ F_body, 接触判断: F_world[2] > 阈值 (地面法向力向上)
            tau_fb = foot_senser_arr.reshape(3, 4)
            foot_force_z = np.zeros(4)
            for i in range(4):
                Ji = J[:, 3*i:3*i+3]  # 3x3 单腿雅可比
                try:
                    F_body = np.linalg.solve(Ji.T, tau_fb[:, i])
                    F_world = R @ F_body
                    foot_force_z[i] = F_world[2]
                except np.linalg.LinAlgError:
                    foot_force_z[i] = 0.0
            foot_contact = (foot_force_z > 1.0).astype(float)

        if self.state == 0:
            # ===== 站立阶段 =====
            flag = np.array([1, 1, 1, 1], dtype=np.float64)
            self.timer += 1

            # QP 力分配
            self.f = qp_standing(rbf_world, b_control, flag)

            for i in range(4):
                tao[:, i] = -J[:, 3*i:3*i+3].T @ R.T @ self.f[3*i:3*i+3]

            if self.timer > 1000:
                self.state = 1
                self.timer = 0
                self.firstswing = np.array([0, 1, 1, 0], dtype=np.float64)
                # 初始化所有腿的 pf_init 为当前足端位置, 避免切换瞬间
                # firstswing=0 的腿 (0,3) 用 zeros 计算贝塞尔轨迹导致加速度异常
                for i in range(4):
                    self.pf_init[:, i] = np.array([x, y, z]) + rbf_world[:, i]
                    self.pf_init[2, i] = 0.015

        elif self.state == 1:
            # ===== 行走阶段 =====
            for i in range(4):
                if self.firstswing[i] == 1:
                    self.swingTimeRemaining[i] = self.swingtime
                    self.pf_init[:, i] = np.array([x, y, z]) + rbf_world[:, i]
                    self.pf_init[2, i] = 0.015
                    self.wf = 0.5 * self.stancetime * np.array([v[0], v[1], 0])
                    if self.wf[0] > 0.35:
                        self.wf[0] = 0.35
                else:
                    self.swingTimeRemaining[i] -= self.dt

                if self.swingTimeRemaining[i] < 0:
                    self.swingTimeRemaining[i] = 0

                # yaw 角速度补偿: 预测摆动期内机体旋转, 落足点预偏转
                theta_yaw = v_yaw * self.swingTimeRemaining[i]
                c_yaw, s_yaw = np.cos(theta_yaw), np.sin(theta_yaw)
                Rz_yaw = np.array([[c_yaw, -s_yaw, 0],
                                   [s_yaw,  c_yaw, 0],
                                   [0,      0,     1]])
                self.pf_final[:, i] = (np.array([x, y, z])
                                       + v_ref * self.swingTimeRemaining[i]
                                       + R @ Rz_yaw @ self.offset[:, i] + self.wf)
                self.pf_final[2, i] = 0.015

            iteration, phase = set_iterations(
                self.n_iterations, self.iterationcounter, self.iterations_between_mpc)

            swingstate = get_swing_state(phase, offsets_float, durations_float)

            mpctable = get_mpc_table(iteration, self.n_iterations, offsets, durations)

            if self.iterationcounter % self.iterations_between_mpc == 0:
                # 姿态误差: 用欧拉角差替代旋转向量, 避免 yaw 跨越 ±π 时不连续
                # 注意符号: 原始 QP 中 eta 的 g 项为 +Q@eta, 其余状态为 -Q@xd
                # 故 eta_des 应取 R→Rd 方向 (与 pos/vel 反号), 即期望-当前
                eta_des_vec = np.array([d_roll - roll_cur,
                                        d_pitch - pitch_cur,
                                        0.0])

                # 构造期望轨迹 (12维: [p, v, η, ω])
                for i in range(self.horizon):
                    if i == 0:
                        px = x + self.dt_mpc * v_ref[0]
                        py = y + self.dt_mpc * v_ref[1]
                        pz = bodyheight + self.dt_mpc * v_ref[2]
                    else:
                        px = trajAll[12*(i-1) + 0] + self.dt_mpc * v_ref[0]
                        py = trajAll[12*(i-1) + 1] + self.dt_mpc * v_ref[1]
                        pz = trajAll[12*(i-1) + 2] + self.dt_mpc * v_ref[2]
                    trajAll[12*i + 0] = px
                    trajAll[12*i + 1] = py
                    trajAll[12*i + 2] = pz
                    trajAll[12*i + 3] = v_ref[0]
                    trajAll[12*i + 4] = v_ref[1]
                    trajAll[12*i + 5] = v_ref[2]
                    trajAll[12*i + 6] = eta_des_vec[0]
                    trajAll[12*i + 7] = eta_des_vec[1]
                    trajAll[12*i + 8] = eta_des_vec[2]
                    trajAll[12*i + 9] = w_ref[0]
                    trajAll[12*i + 10] = w_ref[1]
                    trajAll[12*i + 11] = w_ref[2]

                # 当前状态 (12维: [p, v, η=0, ω], 操作点 η=0)
                self.x_s = np.concatenate([
                    [x, y, z],
                    v,
                    [0, 0, 0],
                    w,
                ])

                # 期望状态
                for i in range(self.horizon):
                    for j in range(12):
                        self.x_desire[12*i + j] = trajAll[12*i + j]

                # MPC 求解 (李群: 传 R, w, fop=前12维操作点力, I, x_op=COM位置)
                fop = self.f[:12] if self.f.size >= 12 else None
                x_op = np.array([x, y, z], dtype=np.float64)
                self.f = solve_mpc(R, w, fop, rbf_world,
                                   self.x_s, self.x_desire, self.dt_mpc,
                                   self.horizon, mpctable, self.mass, self.I, x_op)

            self.iterationcounter += 1

            # 力/力矩分配
            for i in range(4):
                if swingstate[i] > 0:
                    # 摆动腿: 位置跟踪
                    if self.firstswing[i] == 1:
                        self.firstswing[i] = 0
                    p[:, i], pv[:, i], pa[:, i] = swing_trajectory_bezier(
                        self.pf_init[:, i], self.pf_final[:, i],
                        swingstate[i], self.swingtime, self.height)
                    rsf_des[:, i] = R.T @ (p[:, i] - np.array([x, y, z])) - self.rbs[:, i]
                    vleg_des[:, i] = R.T @ (pv[:, i] - v)
                    v_leg[:, i] = J[:, 3*i:3*i+3] @ w_leg[:, i]
                    # 位置误差饱和: 防止腿被障碍物绊住时 PD 误差无限增大
                    # 导致力矩过载, 关节翻转到另一个 IK 分支 (肘↔膝)
                    err_p = rsf_des[:, i] - rsf_body[:, i]
                    err_p_norm = np.linalg.norm(err_p)
                    err_p_max = 0.05  # 5cm 误差饱和
                    if err_p_norm > err_p_max:
                        err_p = err_p * (err_p_max / err_p_norm)
                    Fex = (self.Kp_cartesian @ err_p
                           + self.Kd_cartesian @ (vleg_des[:, i] - v_leg[:, i]))
                    tao_swing = J[:, 3*i:3*i+3].T @ Fex
                    # 关节力矩限幅: 防止拖拽障碍物时翻转
                    tao_max = 8.0
                    tao_norm = np.linalg.norm(tao_swing)
                    if tao_norm > tao_max:
                        tao_swing = tao_swing * (tao_max / tao_norm)
                    tao[:, i] = tao_swing
                elif foot_contact[i] > 0.5:
                    # 支撑腿: 触摸传感器确认接触
                    self.firstswing[i] = 1
                    tao[:, i] = -J[:, 3*i:3*i+3].T @ R.T @ self.f[3*i:3*i+3]
                else:
                    # 未接触但应支撑 (继续压腿)
                    self.firstswing[i] = 1
                    tao[:, i] = -J[:, 3*i:3*i+3].T @ R.T @ self.f[3*i:3*i+3]

        tao1 = tao[:, 0]
        tao2 = tao[:, 1]
        tao3 = tao[:, 2]
        tao4 = tao[:, 3]
        return tao1, tao2, tao3, tao4
