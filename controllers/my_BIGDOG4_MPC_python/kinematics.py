import numpy as np


def skew(r):
    """构造反对称矩阵 (对应 MATLAB Skew.m)"""
    return np.array([
        [0.0, -r[2], r[1]],
        [r[2], 0.0, -r[0]],
        [-r[1], r[0], 0.0],
    ])


def matrix_log_rot(R):
    """旋转矩阵的对数映射, 返回旋转向量 (对应 MATLAB matrixLogRot.m)"""
    tmp = (R[0, 0] + R[1, 1] + R[2, 2] - 1) / 2.0
    if tmp >= 1:
        theta = 0.0
    elif tmp <= -1:
        theta = 3.1415926
    else:
        theta = np.arccos(tmp)

    omega = np.array([R[2, 1] - R[1, 2],
                      R[0, 2] - R[2, 0],
                      R[1, 0] - R[0, 1]])
    if theta > 10e-5:
        omega = omega * theta / (2 * np.sin(theta))
    else:
        omega = omega / 2.0
    return omega


def forward_kinematics(q):
    """
    正运动学: 由关节角计算足端位置 (对应 MATLAB forwardKinematics.m)
    输入 q: 12x1 向量 [FR_hip, FR_leg, FR_foot, FL_hip, FL_leg, FL_foot,
                        BR_hip, BR_leg, BR_foot, BL_hip, BL_leg, BL_foot]
    返回: rsf (3x4 足端相对肩部), rbf (3x4 足端相对躯干中心)
    """
    # Leg 1 - FR
    rsf_x1 = (0.082 * np.sin(q[1]) - 0.175826 * np.cos(q[1])
              + 0.192981 * np.cos(q[1] + q[2]) - 0.00268261 * np.sin(q[1] + q[2]) + 0.065)
    rsf_y1 = (0.082 * np.cos(q[1]) * np.sin(q[0]) - 0.085 * np.cos(q[0])
              + 0.175826 * np.sin(q[0]) * np.sin(q[1])
              - 0.00268261 * np.cos(q[1] + q[2]) * np.sin(q[0])
              - 0.192981 * np.sin(q[0]) * np.sin(q[1] + q[2]))
    rsf_z1 = (0.00268261 * np.cos(q[0]) * np.cos(q[1] + q[2])
              - 0.082 * np.cos(q[0]) * np.cos(q[1])
              - 0.175826 * np.cos(q[0]) * np.sin(q[1])
              - 0.085 * np.sin(q[0])
              + 0.192981 * np.cos(q[0]) * np.sin(q[1] + q[2]))

    # Leg 2 - FL
    rsf_x2 = (0.082 * np.sin(q[4]) - 0.175826 * np.cos(q[4])
              + 0.192981 * np.cos(q[4] + q[5]) - 0.00268261 * np.sin(q[4] + q[5]) + 0.065)
    rsf_y2 = (0.085 * np.cos(q[3]) + 0.082 * np.cos(q[4]) * np.sin(q[3])
              + 0.175826 * np.sin(q[3]) * np.sin(q[4])
              - 0.00268261 * np.cos(q[4] + q[5]) * np.sin(q[3])
              - 0.192981 * np.sin(q[3]) * np.sin(q[4] + q[5]))
    rsf_z2 = (0.085 * np.sin(q[3]) - 0.082 * np.cos(q[3]) * np.cos(q[4])
              - 0.175826 * np.cos(q[3]) * np.sin(q[4])
              + 0.00268261 * np.cos(q[3]) * np.cos(q[4] + q[5])
              + 0.192981 * np.cos(q[3]) * np.sin(q[4] + q[5]))

    # Leg 3 - BR
    rsf_x3 = (0.082 * np.sin(q[7]) - 0.175826 * np.cos(q[7])
              + 0.192981 * np.cos(q[7] + q[8]) - 0.00268261 * np.sin(q[7] + q[8]) - 0.065)
    rsf_y3 = (0.082 * np.cos(q[7]) * np.sin(q[6]) - 0.085 * np.cos(q[6])
              + 0.175826 * np.sin(q[6]) * np.sin(q[7])
              - 0.00268261 * np.cos(q[7] + q[8]) * np.sin(q[6])
              - 0.192981 * np.sin(q[6]) * np.sin(q[7] + q[8]))
    rsf_z3 = (0.00268261 * np.cos(q[6]) * np.cos(q[7] + q[8])
              - 0.082 * np.cos(q[6]) * np.cos(q[7])
              - 0.175826 * np.cos(q[6]) * np.sin(q[7])
              - 0.085 * np.sin(q[6])
              + 0.192981 * np.cos(q[6]) * np.sin(q[7] + q[8]))

    # Leg 4 - BL
    rsf_x4 = (0.082 * np.sin(q[10]) - 0.175826 * np.cos(q[10])
              + 0.192981 * np.cos(q[10] + q[11]) - 0.00268261 * np.sin(q[10] + q[11]) - 0.065)
    rsf_y4 = (0.085 * np.cos(q[9]) + 0.082 * np.cos(q[10]) * np.sin(q[9])
              + 0.175826 * np.sin(q[9]) * np.sin(q[10])
              - 0.00268261 * np.cos(q[10] + q[11]) * np.sin(q[9])
              - 0.192981 * np.sin(q[9]) * np.sin(q[10] + q[11]))
    rsf_z4 = (0.085 * np.sin(q[9]) - 0.082 * np.cos(q[9]) * np.cos(q[10])
              - 0.175826 * np.cos(q[9]) * np.sin(q[10])
              + 0.00268261 * np.cos(q[9]) * np.cos(q[10] + q[11])
              + 0.192981 * np.cos(q[9]) * np.sin(q[10] + q[11]))

    rsf = np.array([
        [rsf_x1, rsf_x2, rsf_x3, rsf_x4],
        [rsf_y1, rsf_y2, rsf_y3, rsf_y4],
        [rsf_z1, rsf_z2, rsf_z3, rsf_z4],
    ])
    rbf = np.array([
        [rsf_x1 + 0.139, rsf_x2 + 0.139, rsf_x3 - 0.139, rsf_x4 - 0.139],
        [rsf_y1 - 0.061, rsf_y2 + 0.061, rsf_y3 - 0.061, rsf_y4 + 0.061],
        [rsf_z1, rsf_z2, rsf_z3, rsf_z4],
    ])
    return rsf, rbf


def jacobian_matrix(q):
    """
    雅可比矩阵 (对应 MATLAB JacobianMatrix.m)
    返回 3x12 矩阵, 每 3 列对应一条腿
    """
    # Leg 1 - FR
    J1 = np.array([
        [0.0,
         0.0819823 * np.cos(q[1]) - 0.192981 * np.sin(q[1] + q[2]) - 0.00268261 * np.cos(q[1] + q[2]) + 0.175826 * np.sin(q[1]),
         -0.00268261 * np.cos(q[1] + q[2]) - 0.192981 * np.sin(q[1] + q[2])],
        [0.085 * np.sin(q[0]) + 0.0819823 * np.cos(q[0]) * np.cos(q[1]) + 0.175826 * np.cos(q[0]) * np.sin(q[1]) - 0.00268261 * np.cos(q[1] + q[2]) * np.cos(q[0]) - 0.192981 * np.sin(q[1] + q[2]) * np.cos(q[0]),
         0.175826 * np.cos(q[1]) * np.sin(q[0]) - 0.0819823 * np.sin(q[0]) * np.sin(q[1]) - 0.192981 * np.cos(q[1] + q[2]) * np.sin(q[0]) + 0.00268261 * np.sin(q[1] + q[2]) * np.sin(q[0]),
         0.00268261 * np.sin(q[1] + q[2]) * np.sin(q[0]) - 0.192981 * np.cos(q[1] + q[2]) * np.sin(q[0])],
        [0.0819823 * np.cos(q[1]) * np.sin(q[0]) - 0.085 * np.cos(q[0]) + 0.175826 * np.sin(q[0]) * np.sin(q[1]) - 0.00268261 * np.cos(q[1] + q[2]) * np.sin(q[0]) - 0.192981 * np.sin(q[1] + q[2]) * np.sin(q[0]),
         0.0819823 * np.cos(q[0]) * np.sin(q[1]) - 0.175826 * np.cos(q[0]) * np.cos(q[1]) + 0.192981 * np.cos(q[1] + q[2]) * np.cos(q[0]) - 0.00268261 * np.sin(q[1] + q[2]) * np.cos(q[0]),
         0.192981 * np.cos(q[1] + q[2]) * np.cos(q[0]) - 0.00268261 * np.sin(q[1] + q[2]) * np.cos(q[0])],
    ])

    # Leg 2 - FL
    J2 = np.array([
        [0.0,
         0.0819823 * np.cos(q[4]) - 0.192981 * np.sin(q[4] + q[5]) - 0.00268261 * np.cos(q[4] + q[5]) + 0.175826 * np.sin(q[4]),
         -0.00268261 * np.cos(q[4] + q[5]) - 0.192981 * np.sin(q[4] + q[5])],
        [0.0819823 * np.cos(q[3]) * np.cos(q[4]) - 0.085 * np.sin(q[3]) + 0.175826 * np.cos(q[3]) * np.sin(q[4]) - 0.00268261 * np.cos(q[4] + q[5]) * np.cos(q[3]) - 0.192981 * np.sin(q[4] + q[5]) * np.cos(q[3]),
         0.175826 * np.cos(q[4]) * np.sin(q[3]) - 0.0819823 * np.sin(q[3]) * np.sin(q[4]) - 0.192981 * np.cos(q[4] + q[5]) * np.sin(q[3]) + 0.00268261 * np.sin(q[4] + q[5]) * np.sin(q[3]),
         0.00268261 * np.sin(q[4] + q[5]) * np.sin(q[3]) - 0.192981 * np.cos(q[4] + q[5]) * np.sin(q[3])],
        [0.085 * np.cos(q[3]) + 0.0819823 * np.cos(q[4]) * np.sin(q[3]) + 0.175826 * np.sin(q[3]) * np.sin(q[4]) - 0.00268261 * np.cos(q[4] + q[5]) * np.sin(q[3]) - 0.192981 * np.sin(q[4] + q[5]) * np.sin(q[3]),
         0.0819823 * np.cos(q[3]) * np.sin(q[4]) - 0.175826 * np.cos(q[3]) * np.cos(q[4]) + 0.192981 * np.cos(q[4] + q[5]) * np.cos(q[3]) - 0.00268261 * np.sin(q[4] + q[5]) * np.cos(q[3]),
         0.192981 * np.cos(q[4] + q[5]) * np.cos(q[3]) - 0.00268261 * np.sin(q[4] + q[5]) * np.cos(q[3])],
    ])

    # Leg 3 - BR
    J3 = np.array([
        [0.0,
         0.0819823 * np.cos(q[7]) - 0.192981 * np.sin(q[7] + q[8]) - 0.00268261 * np.cos(q[7] + q[8]) + 0.175826 * np.sin(q[7]),
         -0.00268261 * np.cos(q[7] + q[8]) - 0.192981 * np.sin(q[7] + q[8])],
        [0.085 * np.sin(q[6]) + 0.0819823 * np.cos(q[6]) * np.cos(q[7]) + 0.175826 * np.cos(q[6]) * np.sin(q[7]) - 0.00268261 * np.cos(q[7] + q[8]) * np.cos(q[6]) - 0.192981 * np.sin(q[7] + q[8]) * np.cos(q[6]),
         0.175826 * np.cos(q[7]) * np.sin(q[6]) - 0.0819823 * np.sin(q[6]) * np.sin(q[7]) - 0.192981 * np.cos(q[7] + q[8]) * np.sin(q[6]) + 0.00268261 * np.sin(q[7] + q[8]) * np.sin(q[6]),
         0.00268261 * np.sin(q[7] + q[8]) * np.sin(q[6]) - 0.192981 * np.cos(q[7] + q[8]) * np.sin(q[6])],
        [0.0819823 * np.cos(q[7]) * np.sin(q[6]) - 0.085 * np.cos(q[6]) + 0.175826 * np.sin(q[6]) * np.sin(q[7]) - 0.00268261 * np.cos(q[7] + q[8]) * np.sin(q[6]) - 0.192981 * np.sin(q[7] + q[8]) * np.sin(q[6]),
         0.0819823 * np.cos(q[6]) * np.sin(q[7]) - 0.175826 * np.cos(q[6]) * np.cos(q[7]) + 0.192981 * np.cos(q[7] + q[8]) * np.cos(q[6]) - 0.00268261 * np.sin(q[7] + q[8]) * np.cos(q[6]),
         0.192981 * np.cos(q[7] + q[8]) * np.cos(q[6]) - 0.00268261 * np.sin(q[7] + q[8]) * np.cos(q[6])],
    ])

    # Leg 4 - BL
    J4 = np.array([
        [0.0,
         0.0819823 * np.cos(q[10]) - 0.192981 * np.sin(q[10] + q[11]) - 0.00268261 * np.cos(q[10] + q[11]) + 0.175826 * np.sin(q[10]),
         -0.00268261 * np.cos(q[10] + q[11]) - 0.192981 * np.sin(q[10] + q[11])],
        [0.0819823 * np.cos(q[9]) * np.cos(q[10]) - 0.085 * np.sin(q[9]) + 0.175826 * np.cos(q[9]) * np.sin(q[10]) - 0.00268261 * np.cos(q[10] + q[11]) * np.cos(q[9]) - 0.192981 * np.sin(q[10] + q[11]) * np.cos(q[9]),
         0.175826 * np.cos(q[10]) * np.sin(q[9]) - 0.0819823 * np.sin(q[9]) * np.sin(q[10]) - 0.192981 * np.cos(q[10] + q[11]) * np.sin(q[9]) + 0.00268261 * np.sin(q[10] + q[11]) * np.sin(q[9]),
         0.00268261 * np.sin(q[10] + q[11]) * np.sin(q[9]) - 0.192981 * np.cos(q[10] + q[11]) * np.sin(q[9])],
        [0.085 * np.cos(q[9]) + 0.0819823 * np.cos(q[10]) * np.sin(q[9]) + 0.175826 * np.sin(q[9]) * np.sin(q[10]) - 0.00268261 * np.cos(q[10] + q[11]) * np.sin(q[9]) - 0.192981 * np.sin(q[10] + q[11]) * np.sin(q[9]),
         0.0819823 * np.cos(q[9]) * np.sin(q[10]) - 0.175826 * np.cos(q[9]) * np.cos(q[10]) + 0.192981 * np.cos(q[10] + q[11]) * np.cos(q[9]) - 0.00268261 * np.sin(q[10] + q[11]) * np.cos(q[9]),
         0.192981 * np.cos(q[10] + q[11]) * np.cos(q[9]) - 0.00268261 * np.sin(q[10] + q[11]) * np.cos(q[9])],
    ])

    J = np.hstack([J1, J2, J3, J4])
    return J
