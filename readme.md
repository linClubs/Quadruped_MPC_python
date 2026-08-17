# Quadruped_MPC_python

<div align="center">
<img src="protos/demo.gif" alt="demo">
</div>

---

+ 四足机器人 MPC 控制仿真项目（[Webots-R2023b](https://github.com/cyberbotics/webots/releases/download/R2023b/webots_2023b_amd64.deb) + Python）。 [视频展示](https://www.bilibili.com/video/BV1Gfb86jEGS/?spm_id_from=333.1387.homepage.video_card.click&vd_source=f4b456ba81d8352a8d8b98b1fe694f07)

+ 只用了osqp，numpy，scipy其他均为显式代码 供初学者入门使用 控制实时性暂时难以保证

+ 修改足端接触检测  关节力矩反馈+雅可比

~~~python
# 遥控器 + 触摸传感器
MPC_JOYSTICK=1 MPC_FOOT_DETECT=touch webots worlds/empty.wbt

# 遥控器 + 力矩反推接触检测
MPC_JOYSTICK=1 MPC_FOOT_DETECT=torque webots worlds/empty.wbt
~~~


## 步态切换

| 键 | 步态 | offsets | durations |
|----|------|---------|-----------|
| `U` | Trotting（对角小跑） | [0,5,5,0] | [5,5,5,5] |
| `I` | Bounding（跳跃） | [5,5,0,0] | [4,4,4,4] |
| `O` | Pronking（四脚齐跳） | [0,0,0,0] | [4,4,4,4] |
| `P` | Galloping（飞奔） | [0,2,7,9] | [4,4,4,4] |
| `J` | Standing（站立） | [0,0,0,0] | [10,10,10,10] |
| `K` | Trot Running（小跑快走） | [0,5,5,0] | [4,4,4,4] |
| `L` | Walking（行走） | [5,0,5,0] | [5,5,5,5] |
| `1` | Walk2（3 触地 1 摆动） | [0,3,5,8] | [8,8,8,8] |

## 速度/位置控制

| 键 | 功能 |
|----|------|
| `W` | 前进 vx=0.5 |
| `S` | 后退 vx=-0.3 |
| `A` | 左移 vy=0.3 |
| `D` | 右移 vy=-0.3 |
| `Q` | 左转 v_yaw=0.5（可与 W/S/A/D 同时按下，边走边转） |
| `E` | 右转 v_yaw=-0.5（可与 W/S/A/D 同时按下，边走边转） |
| `T` | 持续加速前进（最高 2.0，5 秒后生效，同时降低身高到 0.25） |

> 注：`vx/vy` 为 body 系速度，控制器内部通过当前旋转矩阵 R 转换到世界系；`v_yaw` 为偏航角速度跟踪，通过 MPC 的 `w_ref/wd` 前馈 + 落足点 yaw 补偿实现，支持持续旋转。

## 姿态调整

| 键 | 功能 |
|----|------|
| `Z` | roll +0.25 |
| `X` | roll -0.25 |
| `C` | pitch +0.25 |
| `V` | pitch -0.25 |
| `B` | yaw +0.25 |
| `N` | yaw -0.25 |

## 自动流程

- `t < 3s`：站立姿态归位
- `3 < t < 5s`：自动起立抬升（vz=0.1）
- `t > 5s`：可手动控制，默认 Trotting 步态



## 接触检测方案对比

| 方案 | 优点 | 缺点 |
|------|------|------|
| TouchSensor | 直接、可靠 | 需要额外硬件，不真实 |
| MPC 预测力 | 无需传感器 | 用预测值判状态，逻辑循环 |
| 力矩反馈+雅可比 | 物理真实，无需额外硬件 | 雅可比奇异时失效（已用 try-except 兜底） |

本项目同时支持 **TouchSensor** 和 **力矩反馈+雅可比** 两种方案，通过环境变量切换：

- **`MPC_FOOT_DETECT=touch`**（默认）：读取 Webots TouchSensor，返回 1/0 二值接触状态
- **`MPC_FOOT_DETECT=torque`**：读取关节力矩反馈，通过雅可比反推足端力 `F = (J^T)^-1 @ tau`，根据 z 方向力判断接触

## 遥控器控制 (Logitech F710)

支持通过 Logitech F710 无线遥控器进行实时速度控制，使用 pygame 读取手柄输入（pygame 安装在 conda gmr 环境，与系统 python 版本一致 3.10.12）。

### 启用方式

```bash
# 遥控器 + 触摸传感器
MPC_JOYSTICK=1 MPC_FOOT_DETECT=touch webots worlds/empty.wbt

# 遥控器 + 力矩反推接触检测
MPC_JOYSTICK=1 MPC_FOOT_DETECT=torque webots worlds/empty.wbt
```

### F710 模式要求

**DirectInput 模式（LED 灯灭）**：按住手柄中央的 LOGI 按钮，LED 灯熄灭。

### 控制映射

**摇杆**（带启动校准 + 死区 0.1 + 指数滤波 α=0.3）：

| 摇杆 | 方向 | 控制 | 量程 |
|------|------|------|------|
| 左摇杆 Y | 上下 | vx 前进/后退 | ±0.6 m/s |
| 左摇杆 X | 左右 | vy 侧移 | ±0.3 m/s |
| 右摇杆 X | 左右 | v_yaw 偏航角速度 | ±1.57 rad/s |

**按钮**（DirectInput 编号）：

| 按钮 | 编号 | 功能 | 对应键盘 |
|------|------|------|----------|
| X(A) | 0 | walk2 步态 | 1 |
| □(X) | 2 | trot 步态 | U |

### 特性

1. **启动校准**：开机时记录摇杆轴零点，消除摇杆机械漂移
2. **死区处理**：10% 死区，避免静止时漂移
3. **平滑过渡**：指数滤波（α=0.3），避免摇杆突变导致失稳
4. **优雅回退**：无遥控器或 pygame 加载失败时自动回退键盘控制
5. **日志标识**：日志末尾 `in=joy`/`in=key` 标识当前输入源

### 诊断

启动时会在 `/tmp/mpc_joy_diag.txt` 写入遥控器诊断信息（pygame 连接状态、轴数、按钮数、校准零点），便于排查问题。

## 姿态误差计算 (避免 yaw 奇异)

### 问题

原代码用旋转向量 `matrix_log_rot(R.T)`（等价于 `vee(logm(R.T))`）计算姿态误差。旋转向量（轴角表示）在总旋转角 θ 接近 ±π 时存在**不连续点**：

```
yaw = 179.9°:  qw = [-0.0003, -0.0312, -3.1240]   (pitch 误差为负)
yaw = 180.1°:  qw = [-0.0003, +0.0312, +3.1240]   (pitch 误差突然变正!)
```

即使将 yaw 分量清零（`qw[2]=0`），**roll/pitch 分量也会整体反号**，导致姿态反馈方向突然反转，变成正反馈，机器人在持续旋转约 180° 后姿态失控。

### 修复

改用旋转矩阵第三行直接提取 roll/pitch：

```python
roll_cur  = np.arctan2(R[2, 1], R[2, 2])
pitch_cur = np.arcsin(-np.clip(R[2, 0], -1.0, 1.0))
qw = np.array([d_roll - roll_cur, d_pitch - pitch_cur, 0.0])
```

`R[2,0]`、`R[2,1]`、`R[2,2]` 只依赖 roll/pitch，**完全不受 yaw 影响**，因此在任意 yaw 下姿态误差都连续稳定，机器人可无限持续旋转。

### 验证

用 v_yaw=2.0 rad/s（约 1.6 秒达到 180°）测试：

```
t=18.950  yaw=+3.136 (+180°)  roll=0.033  pitch=0.037   ← 即将跨越
t=19.000  yaw=-3.128 (-180°)  roll=0.013  pitch=0.013   ← 跨越! 连续!
t=22.750  yaw=-1.094          roll=0.002  pitch=-0.001  ← 转过 270°, 稳定
```


## 参考

[Quadruped_MPC_matlab](https://github.com/BAO162/Quadruped_MPC_matlab)

