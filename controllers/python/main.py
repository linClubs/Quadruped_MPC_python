"""Representation-Free Model Predictive Control - Python main script.

Python translation of MAIN.m.

Usage:
    /home/lin/software/miniconda3/envs/gmr/bin/python main.py
"""

import os
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt

# Make sure the local packages can be imported when run from the python/ dir.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fcns.params import get_params
from fcns.dynamics import dynamics_SRB
from fcns.fsm import FSMSimple, FSMBound
from fcns.visualization import fig_animate, fig_plot_robot, fig_plot_robot_d
from fcns_mpc.mpc import fcn_get_QP_form_eta, fcn_get_disturbance
from fcns_mpc.ref_traj import fcn_bound_ref_traj, fcn_gen_XdUd

try:
    import quadprog
    HAS_QUADPROG = True
except ImportError:
    HAS_QUADPROG = False


def solve_qp(H, g, Aineq, bineq, Aeq, beq):
    """Solve the QP

        min 0.5 z' H z + g' z
        s.t. Aeq z = beq, Aineq z <= bineq

    using the ``quadprog`` package. Falls back to scipy.optimize.minimize if
    quadprog is unavailable.
    """
    H_sym = 0.5 * (H + H.T)  # ensure symmetry
    if HAS_QUADPROG:
        # quadprog solves: min 0.5 x' G x - a' x s.t. C' x >= b
        G = H_sym
        a = -g
        n_eq = Aeq.shape[0]
        C = np.hstack([Aeq.T, -Aineq.T])
        b = np.concatenate([beq, -bineq])
        try:
            z, _obj, _x, _lag, _iact = quadprog.solve_qp(G, a, C, b, meq=n_eq)
            return z
        except Exception:
            pass  # fall through to scipy

    # Fallback: scipy.optimize.minimize with SLSQP
    from scipy.optimize import minimize
    n = H.shape[0]

    def obj(z):
        return 0.5 * z @ H_sym @ z + g @ z

    def obj_grad(z):
        return H_sym @ z + g

    constraints = []
    if Aeq.shape[0] > 0:
        constraints.append({'type': 'eq', 'fun': lambda z: Aeq @ z - beq, 'jac': lambda z: Aeq})
    if Aineq.shape[0] > 0:
        constraints.append({'type': 'ineq', 'fun': lambda z: bineq - Aineq @ z, 'jac': lambda z: -Aineq})

    res = minimize(obj, np.zeros(n), jac=obj_grad, method='SLSQP', constraints=constraints,
                   options={'maxiter': 200, 'ftol': 1e-9})
    return res.x


def main_live():
    """实时仿真：每步迭代后立即刷新 3D 机器人图。

    与 ``main()`` 相比，不录制 mp4，而是边算边显示，便于观察 MPC 在线行为。
    """
    gait = 0
    p = get_params(gait)
    p['playSpeed'] = 1
    p['flag_movie'] = 0
    p['p_ext'] = np.zeros(3)

    dt_sim = p['simTimeStep']
    SimTimeDuration = 0.5
    MAX_ITER = int(np.floor(SimTimeDuration / p['simTimeStep']))

    p['acc_d'] = 1.0
    p['vel_d'] = np.array([0.5, 0.0])
    p['yaw_d'] = 0.0

    if gait == 1:
        p, Xt, Ut = fcn_bound_ref_traj(p)
        fsm = FSMBound(Xt, p)
    else:
        Xt, Ut = fcn_gen_XdUd(np.array([0.0]), [], np.ones((4, 1)), p)
        fsm = FSMSimple(Xt, p)

    Xt = Xt.astype(float).reshape(-1)
    Ut = Ut.astype(float).reshape(-1)

    # 实时画布：3D 主视图 + 状态子图
    plt.ion()
    fig = plt.figure(figsize=(13, 7), facecolor='white')
    ax_main = fig.add_subplot(2, 3, 1, projection='3d')
    ax_x = fig.add_subplot(2, 3, 2)
    ax_dx = fig.add_subplot(2, 3, 3)
    ax_w = fig.add_subplot(2, 3, 4)
    ax_u = fig.add_subplot(2, 3, 5)
    ax_err = fig.add_subplot(2, 3, 6)

    # 预创建 2D line 对象（用空数据初始化，后续用 set_data 复用）
    empty = np.array([])
    lines_x = [ax_x.plot(empty, empty, c, linewidth=1)[0] for c in ['r', 'g', 'b', 'r', 'g', 'b']]
    for ln in lines_x[3:]:
        ln.set_linestyle('--')
    ax_x.set_title('Position [m]')
    lines_dx = [ax_dx.plot(empty, empty, c, linewidth=1)[0] for c in ['r', 'g', 'b', 'r', 'g', 'b']]
    for ln in lines_dx[3:]:
        ln.set_linestyle('--')
    ax_dx.set_title('Velocity [m/s]')
    lines_w = [ax_w.plot(empty, empty, c, linewidth=1)[0] for c in ['r', 'g', 'b', 'r', 'g', 'b']]
    for ln in lines_w[3:]:
        ln.set_linestyle('--')
    ax_w.set_title('Angular velocity [rad/s]')
    lines_u = [ax_u.plot(empty, empty, c, linewidth=1)[0] for c in ['r', 'g', 'b', 'k', 'r', 'g', 'b', 'k']]
    for ln in lines_u[4:]:
        ln.set_linestyle('--')
    ax_u.set_title('Fz [N]')
    lines_err = [ax_err.plot(empty, empty, c, linewidth=1)[0] for c in ['r', 'g', 'b']]
    ax_err.axhline(0, color='k', linewidth=0.5)
    ax_err.set_title('Tracking error [m]')
    fig.tight_layout()

    t_log = []
    X_log = []
    Xd_log = []
    U_log = []
    Ud_log = []

    tstart = 0.0
    tend = dt_sim
    print(f'Live simulating... (MAX_ITER={MAX_ITER})')
    t0 = time.time()
    for ii in range(MAX_ITER):
        t_ = dt_sim * ii + p['Tmpc'] * np.arange(p['predHorizon'])

        FSM, Xd, Ud, Xt = fsm(t_, Xt, p)
        FSM = np.atleast_1d(FSM)
        Xd = np.atleast_2d(Xd)
        Ud = np.atleast_2d(Ud)

        H, g, Aineq, bineq, Aeq, beq = fcn_get_QP_form_eta(Xt, Ut, Xd, Ud, p)
        z = solve_qp(H, g, Aineq, bineq, Aeq, beq)
        Ut = Ut + z[0:12]

        u_ext, p_ext = fcn_get_disturbance(tstart, p)
        p['p_ext'] = p_ext
        u_ext = 0.0 * u_ext

        sol = solve_ivp(
            lambda t, X: dynamics_SRB(t, X, Ut, Xd[:, 0], 0.0 * u_ext, p),
            [tstart, tend], Xt, method='RK45', rtol=1e-6, atol=1e-9,
        )
        Xt = sol.y.T[-1, :].copy()

        # 记录
        t_log.append(tstart)
        X_log.append(Xt.copy())
        Xd_log.append(Xd[:, 0].copy())
        U_log.append(Ut.copy())
        Ud_log.append(Ud[:, 0].copy())

        # ---- 实时刷新（优化：复用 artist，3D 降频）----
        # 3D 机器人图：每 5 步清空重画一次（cla 是 3D 最慢的操作）
        if ii % 5 == 0 or ii == MAX_ITER - 1:
            ax_main.cla()
            pcom = Xt[0:3]
            ax_main.set_xlim(pcom[0] - 0.5, pcom[0] + 0.5)
            ax_main.set_ylim(pcom[1] - 0.5, pcom[1] + 0.5)
            ax_main.set_zlim(-0.2, 0.6)
            ax_main.set_box_aspect((1, 1, 1))
            ax_main.view_init(elev=15, azim=-60)
            fig_plot_robot(ax_main, Xt, Ut, np.zeros(3), p)
            fig_plot_robot_d(ax_main, Xd[:, 0], np.zeros(12), p)
            ax_main.text(pcom[0], pcom[1], 0.45, f't = {tstart:.2f}s', fontsize=10)
            ax_main.set_title(f'iter {ii + 1}/{MAX_ITER}')

        # 2D 曲线：用 set_data 复用 line 对象（微秒级）
        ta = np.array(t_log)
        Xa = np.array(X_log)
        Xda = np.array(Xd_log)
        Ua = np.array(U_log)
        Uda = np.array(Ud_log)

        lines_x[0].set_data(ta, Xa[:, 0]); lines_x[1].set_data(ta, Xa[:, 1]); lines_x[2].set_data(ta, Xa[:, 2])
        lines_x[3].set_data(ta, Xda[:, 0]); lines_x[4].set_data(ta, Xda[:, 1]); lines_x[5].set_data(ta, Xda[:, 2])
        ax_x.set_xlim(ta[0], ta[-1] + dt_sim)
        ax_x.set_ylim(Xa[:, :3].min() - 0.05, Xa[:, :3].max() + 0.05)

        lines_dx[0].set_data(ta, Xa[:, 3]); lines_dx[1].set_data(ta, Xa[:, 4]); lines_dx[2].set_data(ta, Xa[:, 5])
        lines_dx[3].set_data(ta, Xda[:, 3]); lines_dx[4].set_data(ta, Xda[:, 4]); lines_dx[5].set_data(ta, Xda[:, 5])
        ax_dx.set_xlim(ta[0], ta[-1] + dt_sim)
        vstack = np.vstack([Xa[:, 3:6], Xda[:, 3:6]])
        ax_dx.set_ylim(vstack.min() - 0.1, vstack.max() + 0.1)

        lines_w[0].set_data(ta, Xa[:, 15]); lines_w[1].set_data(ta, Xa[:, 16]); lines_w[2].set_data(ta, Xa[:, 17])
        lines_w[3].set_data(ta, Xda[:, 15]); lines_w[4].set_data(ta, Xda[:, 16]); lines_w[5].set_data(ta, Xda[:, 17])
        ax_w.set_xlim(ta[0], ta[-1] + dt_sim)
        wstack = np.vstack([Xa[:, 15:18], Xda[:, 15:18]])
        ax_w.set_ylim(wstack.min() - 0.5, wstack.max() + 0.5)

        lines_u[0].set_data(ta, Ua[:, 2]); lines_u[1].set_data(ta, Ua[:, 5])
        lines_u[2].set_data(ta, Ua[:, 8]); lines_u[3].set_data(ta, Ua[:, 11])
        lines_u[4].set_data(ta, Uda[:, 2]); lines_u[5].set_data(ta, Uda[:, 5])
        lines_u[6].set_data(ta, Uda[:, 8]); lines_u[7].set_data(ta, Uda[:, 11])
        ax_u.set_xlim(ta[0], ta[-1] + dt_sim)
        ustack = np.vstack([Ua[:, [2, 5, 8, 11]], Uda[:, [2, 5, 8, 11]]])
        ax_u.set_ylim(ustack.min() - 5, ustack.max() + 5)

        err = Xa - Xda
        lines_err[0].set_data(ta, err[:, 0]); lines_err[1].set_data(ta, err[:, 1]); lines_err[2].set_data(ta, err[:, 2])
        ax_err.set_xlim(ta[0], ta[-1] + dt_sim)
        ax_err.set_ylim(err[:, :3].min() - 0.02, err[:, :3].max() + 0.02)

        # 用 draw_idle 替代 pause 的强制重绘，更轻量
        fig.canvas.draw_idle()
        plt.pause(0.001)  # 让出 GUI 事件循环

        tstart = tend
        tend = tstart + dt_sim

    print(f'Live simulation complete in {time.time() - t0:.2f}s')
    plt.ioff()
    plt.show()
    return np.array(t_log), np.array(X_log), np.array(Xd_log)


def main():
    # --- parameters ---
    gait = 0
    p = get_params(gait)
    p['playSpeed'] = 1
    p['flag_movie'] = 1
    use_qpSWIFT = 1  # 0 -> quadprog, 1 -> qpSWIFT (we still use quadprog here)

    dt_sim = p['simTimeStep']
    SimTimeDuration = 0.5  # [sec]
    MAX_ITER = int(np.floor(SimTimeDuration / p['simTimeStep']))

    # desired trajectory
    p['acc_d'] = 1.0
    p['vel_d'] = np.array([0.5, 0.0])
    p['yaw_d'] = 0.0

    # --- initial condition ---
    if gait == 1:
        p, Xt, Ut = fcn_bound_ref_traj(p)
        fsm = FSMBound(Xt, p)
    else:
        Xt, Ut = fcn_gen_XdUd(np.array([0.0]), [], np.ones((4, 1)), p)
        fsm = FSMSimple(Xt, p)

    Xt = Xt.astype(float).reshape(-1)
    Ut = Ut.astype(float).reshape(-1)

    # --- logging ---
    tstart = 0.0
    tend = dt_sim

    tout = np.zeros((0, 1))
    Xout = np.zeros((0, Xt.size))
    Uout = np.zeros((0, Ut.size))
    Xdout = np.zeros((0, Xt.size))
    Udout = np.zeros((0, Ut.size))
    Uext_log = np.zeros((0, 3))
    FSMout = np.zeros((0, 4))

    print(f'Calculating... (MAX_ITER={MAX_ITER})')
    t0 = time.time()
    for ii in range(MAX_ITER):
        t_ = dt_sim * ii + p['Tmpc'] * np.arange(p['predHorizon'])

        # FSM
        if gait == 1:
            FSM, Xd, Ud, Xt = fsm(t_, Xt, p)
        else:
            FSM, Xd, Ud, Xt = fsm(t_, Xt, p)
        FSM = np.atleast_1d(FSM)
        Xd = np.atleast_2d(Xd)
        Ud = np.atleast_2d(Ud)

        # MPC: build and solve QP
        H, g, Aineq, bineq, Aeq, beq = fcn_get_QP_form_eta(Xt, Ut, Xd, Ud, p)
        z = solve_qp(H, g, Aineq, bineq, Aeq, beq)
        Ut = Ut + z[0:12]

        # External disturbance
        u_ext, p_ext = fcn_get_disturbance(tstart, p)
        p['p_ext'] = p_ext
        u_ext = 0.0 * u_ext  # disabled, mirrors MAIN.m

        # Simulate
        sol = solve_ivp(
            lambda t, X: dynamics_SRB(t, X, Ut, Xd[:, 0], 0.0 * u_ext, p),
            [tstart, tend], Xt, method='RK45', rtol=1e-6, atol=1e-9,
        )
        t_ode = sol.t
        X_ode = sol.y.T  # (n_t, 30)

        # Update state
        Xt = X_ode[-1, :].copy()
        tstart = tend
        tend = tstart + dt_sim

        # Log (skip the first sample which duplicates the previous end)
        if t_ode.size > 1:
            tsamp = t_ode[1:].reshape(-1, 1)
            Xsamp = X_ode[1:, :]
        else:
            tsamp = np.array([[t_ode[-1]]])
            Xsamp = X_ode[-1:, :]

        tout = np.vstack([tout, tsamp])
        Xout = np.vstack([Xout, Xsamp])
        Uout = np.vstack([Uout, np.tile(Ut.reshape(1, -1), (Xsamp.shape[0], 1))])
        Xdout = np.vstack([Xdout, np.tile(Xd[:, 0].reshape(1, -1), (Xsamp.shape[0], 1))])
        Udout = np.vstack([Udout, np.tile(Ud[:, 0].reshape(1, -1), (Xsamp.shape[0], 1))])
        Uext_log = np.vstack([Uext_log, np.tile(u_ext.reshape(1, -1), (Xsamp.shape[0], 1))])
        if FSM.size == 1:
            FSMrow = np.array([FSM[0], FSM[0], FSM[0], FSM[0]])
        else:
            FSMrow = FSM.reshape(-1)
        FSMout = np.vstack([FSMout, np.tile(FSMrow.reshape(1, -1), (Xsamp.shape[0], 1))])

        if (ii + 1) % 10 == 0 or ii == MAX_ITER - 1:
            print(f'  iter {ii + 1}/{MAX_ITER}')

    print(f'Calculation complete in {time.time() - t0:.2f}s')

    # --- Animation ---
    save_path = 'test.mp4' if p['flag_movie'] else None
    t_anim, EA, EAd = fig_animate(tout.flatten(), Xout, Uout, Xdout, Udout, Uext_log, p, save_path=save_path)
    return t_anim, EA, EAd


if __name__ == '__main__':
    # 用法：
    #   python main.py          # 默认：离线仿真 + 录制 test.mp4
    #   python main.py live     # 实时仿真：边算边显示
    #   python main.py offline  # 离线仿真 + 录制 test.mp4
    mode = sys.argv[1] if len(sys.argv) > 1 else 'offline'
    if mode == 'live':
        main_live()
    else:
        main()
