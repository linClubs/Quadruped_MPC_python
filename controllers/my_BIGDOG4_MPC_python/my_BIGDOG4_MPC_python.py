#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quadruped MPC Python 控制器 (对应 MATLAB my_BIGDOG4_MPC.m)
项目地址: https://github.com/BAO162/Quadruped_MPC_matlab.git
"""
import sys
import os
import numpy as np

# 添加 Webots Python API 路径
# WEBOTS_PYTHON_PATH = "/usr/local/webots/lib/controller/python"
# if WEBOTS_PYTHON_PATH not in sys.path:
#     sys.path.insert(0, WEBOTS_PYTHON_PATH)

# # 添加 Webots 库路径 (Linux)
# WEBOTS_LIB_PATH = "/usr/local/webots/lib/controller"
# if os.path.exists(WEBOTS_LIB_PATH):
#     os.environ["LD_LIBRARY_PATH"] = (
#         WEBOTS_LIB_PATH + ":" + os.environ.get("LD_LIBRARY_PATH", "")
#     )

from controller import Robot

from mpc_controller import MPCController


def rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rot_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


# pygame 遥控器支持 (装在 conda gmr 环境, 版本 3.10.12 与系统 python 一致)
# 启用条件: 环境变量 MPC_JOYSTICK=1
_pygame_ok = False
try:
    if os.environ.get("MPC_JOYSTICK", "0") == "1":
        import pygame
        pygame.init()
        pygame.joystick.init()
        _pygame_ok = True
except Exception as _e:
    _pygame_ok = False
    _pygame_err = str(_e)


class PygameJoystick:
    """基于 pygame 的遥控器封装.

    pygame.joystick 轴值已归一化到 [-1.0, 1.0]; 摇杆 Y 轴上推为负.
    需在主循环中定期调用 poll() 刷新事件 (内部调用 pygame.event.pump).
    """

    def __init__(self, joy_id=0):
        self.joy = None
        self.connected = False
        self.n_axes = 0
        self.n_buttons = 0
        self.axes_zero = {}  # 校准零点 (归一化值)
        try:
            if pygame.joystick.get_count() > joy_id:
                self.joy = pygame.joystick.Joystick(joy_id)
                self.joy.init()
                self.n_axes = self.joy.get_numaxes()
                self.n_buttons = self.joy.get_numbuttons()
                self.connected = True
                # 校准: 摇杆轴 0-3 取当前为零点
                pygame.event.pump()
                for ax in range(min(4, self.n_axes)):
                    self.axes_zero[ax] = self.joy.get_axis(ax)
        except Exception:
            self.connected = False
            self.joy = None

    def poll(self):
        """刷新 pygame 事件队列, 必须在主循环定期调用."""
        pygame.event.pump()

    def getAxisCalibrated(self, ax_id):
        """读取归一化轴值并减去校准零点, 限幅到 [-1, 1]."""
        if self.joy is None or ax_id >= self.n_axes:
            return 0.0
        val = self.joy.get_axis(ax_id) - self.axes_zero.get(ax_id, 0.0)
        return max(-1.0, min(1.0, val))

    def getButton(self, btn_id):
        if self.joy is None or btn_id >= self.n_buttons:
            return 0
        return self.joy.get_button(btn_id)

    def isConnected(self):
        return self.connected


def main():
    robot = Robot()
    time_step = 2  # ms

    # ===== 参数 =====
    bodyheight = 0.3
    vx = 0.0
    vy = 0.0
    v_yaw = 0.0
    vz = 0.1
    d_roll = 0.0
    d_pitch = 0.0
    d_yaw = 0.0

    rbs = np.array([
        [0.139, 0.139, -0.139, -0.139],
        [-0.061, 0.061, -0.061, 0.061],
        [0, 0, 0, 0],
    ])
    mass = 13.5
    I = np.diag([0.06150, 0.1313, 0.1646])
    
    offset = np.array([
        [0.204, 0.204, -0.204, -0.204],
        [-0.146, 0.146, -0.146, 0.146],
        [0, 0, 0, 0],
    ])

    Kpcom = np.diag([400, 400, 400])
    Kdcom = np.diag([160, 160, 160])
    Kpbase = np.diag([1000, 1000, 1000])
    Kdbase = np.diag([40, 40, 40])

    dt = 0.002
    iterations_between_mpc = 15
    stancetime = 0.15
    swingtime = 0.15
    height = 0.07
    horizon = 10
    Kp_cartesian = np.diag([100, 100, 100])
    Kd_cartesian = np.diag([10, 20, 10])

    n_iterations = 10
    offsets = np.array([0, 5, 5, 0], dtype=np.float64)
    durations = np.array([5, 5, 5, 5], dtype=np.float64)

    # ===== 创建 MPC 控制器 =====
    mpc = MPCController(
        rbs=rbs, mass=mass, I=I, offset=offset,
        Kpcom=Kpcom, Kdcom=Kdcom, Kpbase=Kpbase, Kdbase=Kdbase,
        dt=dt, iterations_between_mpc=iterations_between_mpc,
        stancetime=stancetime, swingtime=swingtime, height=height,
        horizon=horizon, Kp_cartesian=Kp_cartesian, Kd_cartesian=Kd_cartesian,
        vz=vz, n_iterations=n_iterations,
    )

    # ===== 传感器 =====
    gyro = robot.getDevice("gyro")
    gyro.enable(time_step)
    imu = robot.getDevice("imu")
    imu.enable(time_step)
    gps = robot.getDevice("gps")
    gps.enable(time_step)
    accelerometer = robot.getDevice("accelerometer")
    accelerometer.enable(time_step)

    # ===== 电机 =====
    motor_names = [
        "FR_hip_motor", "FR_leg_motor", "FR_foot_motor",
        "FL_hip_motor", "FL_leg_motor", "FL_foot_motor",
        "BR_hip_motor", "BR_leg_motor", "BR_foot_motor",
        "BL_hip_motor", "BL_leg_motor", "BL_foot_motor",
    ]
    motors = {}
    for name in motor_names:
        m = robot.getDevice(name)
        m.setPosition(float("inf"))
        m.setVelocity(1.0)
        m.enableTorqueFeedback(time_step)
        motors[name] = m

    # ===== 位置传感器 =====
    position_sensor_names = [
        "FR_hip_position_sensor", "FR_leg_position_sensor", "FR_foot_position_sensor",
        "FL_hip_position_sensor", "FL_leg_position_sensor", "FL_foot_position_sensor",
        "BR_hip_position_sensor", "BR_leg_position_sensor", "BR_foot_position_sensor",
        "BL_hip_position_sensor", "BL_leg_position_sensor", "BL_foot_position_sensor",
    ]
    position_sensors = {}
    for name in position_sensor_names:
        ps = robot.getDevice(name)
        ps.enable(time_step)
        position_sensors[name] = ps

    # ===== 足端触摸传感器 (对应 MATLAB FL_TOUCH 等) =====
    # 接触检测方式: "touch"=触摸传感器(默认), "torque"=关节力矩反推
    foot_detect_mode = os.environ.get("MPC_FOOT_DETECT", "touch").lower()
    touch_sensors = {}
    if foot_detect_mode == "touch":
        for prefix in ["FR", "FL", "BR", "BL"]:
            ts = robot.getDevice(f"{prefix}_TOUCH")
            ts.enable(time_step)
            touch_sensors[prefix] = ts
        print(f"[MPC] 足端接触检测: 触摸传感器 (MPC_FOOT_DETECT=touch)")
    else:
        print(f"[MPC] 足端接触检测: 关节力矩反推 (MPC_FOOT_DETECT=torque)")

    # ===== 键盘 =====
    keyboard = robot.getKeyboard()
    keyboard.enable(time_step)

    # ===== 遥控器 (Logitech F710, 可选, 环境变量 MPC_JOYSTICK=1 启用) =====
    # 使用 pygame 读取 (装在 conda gmr 环境, 模块级已初始化)
    # pygame.joystick 轴值已归一化 [-1,1], 内部启动校准消除漂移
    joystick = None
    joystick_diag = []
    if os.environ.get("MPC_JOYSTICK", "0") == "1":
        if _pygame_ok:
            joystick = PygameJoystick(0)
            if joystick.isConnected():
                joystick_diag.append(f"pygame connected=True, axes={joystick.n_axes}, buttons={joystick.n_buttons}")
                joystick_diag.append(f"axes_zero(校准零点)={joystick.axes_zero}")
            else:
                joystick_diag.append("pygame 已加载但未检测到遥控器, 请检查设备")
                joystick = None
        else:
            joystick_diag.append(f"pygame 初始化失败: {_pygame_err}")
        with open("/tmp/mpc_joy_diag.txt", "w") as f:
            f.write("MPC_JOYSTICK=1 诊断:\n")
            for line in joystick_diag:
                f.write(f"  {line}\n")

    # 速度平滑过渡 (指数滤波)
    vx_smooth = 0.0
    vy_smooth = 0.0
    vyaw_smooth = 0.0
    alpha_smooth = 0.3  # 滤波系数 (0~1, 越大跟随越快)

    # ===== 日志文件 =====
    log_file = open("/tmp/mpc_log.txt", "w")

    # ===== 初始状态 =====
    t = 0.0
    step_count = 0
    q1 = np.zeros(3)
    q2 = np.zeros(3)
    q3 = np.zeros(3)
    q4 = np.zeros(3)
    tao1 = None
    tao2 = None
    tao3 = None
    tao4 = None
    xyz = np.zeros(3)
    rpy = np.zeros(3)
    xyz_prev = np.zeros(3)

    # ===== 自动测试模式 (环境变量 MPC_AUTO_TEST=1 启用) =====
    auto_test = os.environ.get("MPC_AUTO_TEST", "0") == "1"

    # ===== 主循环 =====
    while robot.step(time_step) != -1:
        # 读取所有按键 (支持多键同时按下, 如 W+Q 边走边转)
        keys = []
        k = keyboard.getKey()
        while k != -1:
            keys.append(k)
            k = keyboard.getKey()

        # 遥控器输入 (优先级高于键盘, 但低于自动测试)
        # Logitech F710 DirectInput pygame 轴映射 (LED 灭):
        #   0=左摇杆X, 1=左摇杆Y, 2=右摇杆X, 3=右摇杆Y, 4=L2, 5=R2
        # 用户映射: 左LY→vx, 左LX→vy, 右LX→v_yaw
        # pygame Y 轴上推为负, 前推需取反
        joystick_active = False
        if joystick is not None:
            try:
                joystick.poll()  # 刷新 pygame 事件
                ax_left_x  = joystick.getAxisCalibrated(0)  # 左摇杆左右
                ax_left_y  = joystick.getAxisCalibrated(1)  # 左摇杆上下
                ax_right_x = joystick.getAxisCalibrated(3)  # 右摇杆左右
                joystick_active = True
            except Exception:
                joystick_active = False
        if joystick_active:

            # 死区处理 (消除摇杆漂移)
            deadzone = 0.1
            def apply_deadzone(v, dz=deadzone):
                if abs(v) < dz:
                    return 0.0
                # 线性映射死区外部分到 [0,1]
                return (v - dz * np.sign(v)) / (1.0 - dz)

            ax_left_x  = apply_deadzone(ax_left_x)
            ax_left_y  = apply_deadzone(ax_left_y)
            ax_right_x = apply_deadzone(ax_right_x)

            # 速度映射 (摇杆 → 速度指令)
            # 左摇杆 Y: 前推(-) → 前进(+vx), 量程 1.5 m/s
            # 左摇杆 X: 右推(+) → 右移(-vy, body 系), 量程 0.8 m/s
            # 右摇杆 X: 右推(+) → 右转(-v_yaw), 量程 1.0 rad/s
            vx_j   = -ax_left_y  * 0.6
            vy_j   = -ax_left_x  * 0.3
            vyaw_j = -ax_right_x * 1.57

            # 按钮处理: 只保留 1(walk2) 和 U(trot) 两个步态
            # F710 DirectInput: 0=X(A), 2=□(X)
            if joystick.getButton(0) == 1:   # X(A) → walk2
                keys.append(ord("1"))
            if joystick.getButton(2) == 1:   # □(X) → trot
                keys.append(ord("u"))

        # 自动测试: t>=6 切换 walk2, t>=8 前进, t>=12 边走边转
        if auto_test:
            if 6.0 <= t < 8.0:
                keys = [ord("1")]
            elif 8.0 <= t < 12.0:
                keys = [ord("w")]
            elif t >= 12.0:
                keys = [ord("w"), ord("q")]

        d_roll = 0.0
        d_pitch = 0.0
        d_yaw = 0.0

        # 姿态控制
        for key in keys:
            if key == ord("Z") or key == ord("z"):
                d_roll = 0.25
            elif key == ord("X") or key == ord("x"):
                d_roll = -0.25
            elif key == ord("C") or key == ord("c"):
                d_pitch = 0.25
            elif key == ord("V") or key == ord("v"):
                d_pitch = -0.25
            elif key == ord("B") or key == ord("b"):
                d_yaw = 0.25
            elif key == ord("N") or key == ord("n"):
                d_yaw = -0.25

        # 速度控制
        if joystick_active:
            # 遥控器模式: 摇杆值 + 指数滤波平滑过渡
            vx_smooth   = alpha_smooth * vx_j     + (1 - alpha_smooth) * vx_smooth
            vy_smooth   = alpha_smooth * vy_j     + (1 - alpha_smooth) * vy_smooth
            vyaw_smooth = alpha_smooth * vyaw_j   + (1 - alpha_smooth) * vyaw_smooth
            vx = vx_smooth
            vy = vy_smooth
            v_yaw = vyaw_smooth
            bodyheight = 0.3
        elif any(k in (ord("T"), ord("t")) for k in keys):
            # 键盘 T: 自动前进模式
            if t > 5:
                vx = vx + (time_step / 2000.0)
                if vx > 2:
                    vx = 2.0
            bodyheight = 0.25
        else:
            # 键盘模式
            vx = 0.0
            vy = 0.0
            bodyheight = 0.3
            v_yaw = 0.0

            for key in keys:
                if key == ord("W") or key == ord("w"):
                    vx = 0.5
                elif key == ord("S") or key == ord("s"):
                    vx = -0.3
                elif key == ord("A") or key == ord("a"):
                    vy = 0.3
                elif key == ord("D") or key == ord("d"):
                    vy = -0.3

            # 偏航角速度控制 (Q/E 转弯, 可与 W/S/A/D 同时使用)
                if key == ord("Q") or key == ord("q"):
                    v_yaw = 0.5
                elif key == ord("E") or key == ord("e"):
                    v_yaw = -0.5

        # 步态切换
        # 同步 swingtime/stancetime 与实际步态周期匹配
        # 实际摆动 = (n_iterations - duration) * iterations_between_mpc * dt
        # 实际支撑 = duration * iterations_between_mpc * dt
        for key in keys:
            if key == ord("U") or key == ord("u"):
                offsets = np.array([0, 5, 5, 0], dtype=np.float64)
                durations = np.array([5, 5, 5, 5], dtype=np.float64)
                mpc.swingtime = (n_iterations - 5) * iterations_between_mpc * dt
                mpc.stancetime = 5 * iterations_between_mpc * dt
            elif key == ord("I") or key == ord("i"):
                offsets = np.array([5, 5, 0, 0], dtype=np.float64)
                durations = np.array([4, 4, 4, 4], dtype=np.float64)
                mpc.swingtime = (n_iterations - 4) * iterations_between_mpc * dt
                mpc.stancetime = 4 * iterations_between_mpc * dt
            elif key == ord("O") or key == ord("o"):
                offsets = np.array([0, 0, 0, 0], dtype=np.float64)
                durations = np.array([4, 4, 4, 4], dtype=np.float64)
                mpc.swingtime = (n_iterations - 4) * iterations_between_mpc * dt
                mpc.stancetime = 4 * iterations_between_mpc * dt
            elif key == ord("P") or key == ord("p"):
                offsets = np.array([0, 2, 7, 9], dtype=np.float64)
                durations = np.array([4, 4, 4, 4], dtype=np.float64)
                mpc.swingtime = (n_iterations - 4) * iterations_between_mpc * dt
                mpc.stancetime = 4 * iterations_between_mpc * dt
            elif key == ord("J") or key == ord("j"):
                offsets = np.array([0, 0, 0, 0], dtype=np.float64)
                durations = np.array([10, 10, 10, 10], dtype=np.float64)
                mpc.swingtime = (n_iterations - 10) * iterations_between_mpc * dt
                mpc.stancetime = 10 * iterations_between_mpc * dt
            elif key == ord("K") or key == ord("k"):
                offsets = np.array([0, 5, 5, 0], dtype=np.float64)
                durations = np.array([4, 4, 4, 4], dtype=np.float64)
                mpc.swingtime = (n_iterations - 4) * iterations_between_mpc * dt
                mpc.stancetime = 4 * iterations_between_mpc * dt
            elif key == ord("L") or key == ord("l"):
                offsets = np.array([5, 0, 5, 0], dtype=np.float64)
                durations = np.array([5, 5, 5, 5], dtype=np.float64)
                mpc.swingtime = (n_iterations - 5) * iterations_between_mpc * dt
                mpc.stancetime = 5 * iterations_between_mpc * dt
            elif key == ord("1"):
                # Walk2: 3 触地 1 摆动, 摆动持续2步避免双腿同时腾空
                offsets = np.array([0, 3, 5, 8], dtype=np.float64)
                durations = np.array([8, 8, 8, 8], dtype=np.float64)
                mpc.swingtime = (n_iterations - 8) * iterations_between_mpc * dt
                mpc.stancetime = 8 * iterations_between_mpc * dt

        # 保存上一步状态
        q1_u = q1.copy()
        q2_u = q2.copy()
        q3_u = q3.copy()
        q4_u = q4.copy()
        xyz_prev = xyz.copy()

        # 读取传感器
        omega = np.array(gyro.getValues())
        rpy = np.array(imu.getRollPitchYaw())
        xyz = np.array(gps.getValues())

        # 旋转矩阵
        R = rot_z(rpy[2]) @ rot_y(rpy[1]) @ rot_x(rpy[0])

        x, y, z = xyz[0], xyz[1], xyz[2]
        w = omega

        # 读取关节位置
        def read_joint(prefix):
            hip = position_sensors[f"{prefix}_hip_position_sensor"].getValue()
            leg = position_sensors[f"{prefix}_leg_position_sensor"].getValue()
            foot = position_sensors[f"{prefix}_foot_position_sensor"].getValue()
            return np.array([hip, leg + 0.45, foot - 1.40])

        q1 = read_joint("FR")
        q2 = read_joint("FL")
        q3 = read_joint("BR")
        q4 = read_joint("BL")

        # 关节速度
        dt_sec = time_step / 1000.0
        w1 = (q1 - q1_u) / dt_sec
        w2 = (q2 - q2_u) / dt_sec
        w3 = (q3 - q3_u) / dt_sec
        w4 = (q4 - q4_u) / dt_sec

        # 体速度
        v = (xyz - xyz_prev) / dt_sec

        # 读取关节力矩反馈 (用于力矩反推足端接触力)
        tau_fb = np.zeros((3, 4))
        for i, prefix in enumerate(["FR", "FL", "BR", "BL"]):
            tau_fb[0, i] = motors[f"{prefix}_hip_motor"].getTorqueFeedback()
            tau_fb[1, i] = motors[f"{prefix}_leg_motor"].getTorqueFeedback()
            tau_fb[2, i] = motors[f"{prefix}_foot_motor"].getTorqueFeedback()

        # 足端接触数据: touch 模式用触摸传感器, torque 模式用力矩反推
        if foot_detect_mode == "touch":
            foot_senser = np.zeros(4)
            for i, prefix in enumerate(["FR", "FL", "BR", "BL"]):
                foot_senser[i] = touch_sensors[prefix].getValue()
        else:
            # 力矩反推: 传 3x4 关节力矩, 由 MPC 内部转换为接触状态
            foot_senser = tau_fb

        if t < 3:
            # 站立姿态
            for prefix in ["FR", "FL", "BR", "BL"]:
                motors[f"{prefix}_hip_motor"].setPosition(0)
                motors[f"{prefix}_leg_motor"].setPosition(-0.45)
                motors[f"{prefix}_foot_motor"].setPosition(1.4)
                motors[f"{prefix}_hip_motor"].setVelocity(1.0)
                motors[f"{prefix}_leg_motor"].setVelocity(1.0)
                motors[f"{prefix}_foot_motor"].setVelocity(1.0)
        else:
            # MPC 控制
            tao1, tao2, tao3, tao4 = mpc.compute(
                R, w, x, y, z, v, q1, q2, q3, q4,
                w1, w2, w3, w4, t, foot_senser, vx, vy, v_yaw,
                offsets, durations, bodyheight,
                d_roll, d_pitch, d_yaw,
            )

            # 力矩限幅
            tao1 = np.clip(tao1, -30, 30)
            tao2 = np.clip(tao2, -30, 30)
            tao3 = np.clip(tao3, -30, 30)
            tao4 = np.clip(tao4, -30, 30)

            # 施加力矩
            motors["FR_hip_motor"].setTorque(tao1[0])
            motors["FR_leg_motor"].setTorque(tao1[1])
            motors["FR_foot_motor"].setTorque(tao1[2])

            motors["FL_hip_motor"].setTorque(tao2[0])
            motors["FL_leg_motor"].setTorque(tao2[1])
            motors["FL_foot_motor"].setTorque(tao2[2])

            motors["BR_hip_motor"].setTorque(tao3[0])
            motors["BR_leg_motor"].setTorque(tao3[1])
            motors["BR_foot_motor"].setTorque(tao3[2])

            motors["BL_hip_motor"].setTorque(tao4[0])
            motors["BL_leg_motor"].setTorque(tao4[1])
            motors["BL_foot_motor"].setTorque(tao4[2])

        # 日志记录 (每 25 步 = 50ms 记录一次)
        if step_count % 25 == 0:
            # 接触状态摘要: touch 模式直接取值, torque 模式显示力矩模长
            fs = np.asarray(foot_senser)
            if fs.ndim == 1 and fs.size == 4:
                touch_str = f"[{fs[0]:.0f},{fs[1]:.0f},{fs[2]:.0f},{fs[3]:.0f}]"
            else:
                # torque 模式: 显示每条腿关节力矩模长
                tau_norm = np.linalg.norm(fs.reshape(3, 4), axis=0)
                touch_str = f"[{tau_norm[0]:.1f},{tau_norm[1]:.1f},{tau_norm[2]:.1f},{tau_norm[3]:.1f}]"
            log_file.write(
                f"t={t:.3f} xyz=[{x:.3f},{y:.3f},{z:.3f}] "
                f"rpy=[{rpy[0]:.3f},{rpy[1]:.3f},{rpy[2]:.3f}] "
                f"w=[{w[0]:.3f},{w[1]:.3f},{w[2]:.3f}] "
                f"touch={touch_str} "
                f"vx={vx:.2f} vy={vy:.2f} vyaw={v_yaw:.2f} "
                f"mode={'mpc' if t >= 3 else 'stand'} "
                f"in={'joy' if joystick_active else 'key'}\n"
            )
            if t >= 3 and tao1 is not None:
                log_file.write(
                    f"  tao1=[{tao1[0]:.2f},{tao1[1]:.2f},{tao1[2]:.2f}] "
                    f"tao2=[{tao2[0]:.2f},{tao2[1]:.2f},{tao2[2]:.2f}] "
                    f"tao3=[{tao3[0]:.2f},{tao3[1]:.2f},{tao3[2]:.2f}] "
                    f"tao4=[{tao4[0]:.2f},{tao4[1]:.2f},{tao4[2]:.2f}]\n"
                )
            log_file.flush()

        t += time_step / 1000.0
        step_count += 1


if __name__ == "__main__":
    main()
