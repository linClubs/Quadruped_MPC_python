"""Visualization helpers.

Python translation of fcns/fig_animate.m, fcns/fig_plot_robot.m and
fcns/fig_plot_robot_d.m.

Matplotlib's 3D plotting is used instead of MATLAB's fill3/plot3. The API is
necessarily different from MATLAB but the visual content matches.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .maps import hatMap
from .rotations import rx, ry
from .kinematics import fcn_invKin3, fcn_X2EA


def _leg_kin(Twd2com, q, p):
    L = p['L']
    W = p['W']
    d = p['d']
    l1 = p['l1']
    l2 = p['l2']
    sign_L = p['sign_L']
    sign_d = p['sign_d']

    Tcom2h = np.block([
        [rx(q[0]), np.array([[sign_L * L / 2.0, sign_d * W / 2.0, 0.0]]).T],
        [np.zeros((1, 3)), np.array([[1.0]])],
    ])
    Th2s = np.block([
        [ry(q[1]), np.array([[0.0, sign_d * d, 0.0]]).T],
        [np.zeros((1, 3)), np.array([[1.0]])],
    ])
    Ts2k = np.block([
        [ry(q[2]), np.array([[l1, 0.0, 0.0]]).T],
        [np.zeros((1, 3)), np.array([[1.0]])],
    ])
    Tk2f = np.block([
        [np.eye(3), np.array([[l2, 0.0, 0.0]]).T],
        [np.zeros((1, 3)), np.array([[1.0]])],
    ])
    Twd2h = Twd2com @ Tcom2h
    Twd2s = Twd2h @ Th2s
    Twd2k = Twd2s @ Ts2k
    Twd2f = Twd2k @ Tk2f

    p_h_wd = Twd2h[0:3, 3]
    p_s_wd = Twd2s[0:3, 3]
    p_k_wd = Twd2k[0:3, 3]
    p_f_wd = Twd2f[0:3, 3]
    return np.column_stack([p_h_wd, p_s_wd, p_k_wd, p_f_wd])


def _set_signs(p, i_leg):
    if i_leg == 0:
        p['sign_L'] = 1
        p['sign_d'] = 1
    elif i_leg == 1:
        p['sign_L'] = 1
        p['sign_d'] = -1
    elif i_leg == 2:
        p['sign_L'] = -1
        p['sign_d'] = 1
    else:
        p['sign_L'] = -1
        p['sign_d'] = -1


def fig_plot_robot(ax, Xt, Ut, Ue, p):
    """Plot the robot's body, legs, feet and ground reaction forces."""
    L = p['L']
    W = p['W']
    h = p['h']
    body_color = p['body_color']
    leg_color = p['leg_color']
    ground_color = p['ground_color']

    pcom = Xt[0:3].reshape(3, 1)
    dpc = Xt[3:6].reshape(3, 1)
    R = Xt[6:15].reshape(3, 3, order='F')
    wb = Xt[15:18].reshape(3, 1)
    pf34 = Xt[18:30].reshape(3, 4, order='F')

    f34 = Ut.reshape(3, 4, order='F')

    Twd2com = np.block([
        [R, pcom],
        [np.zeros((1, 3)), np.array([[1.0]])],
    ])

    hip_offsets = [
        np.array([L / 2, W / 2, 0.0]),
        np.array([L / 2, -W / 2, 0.0]),
        np.array([-L / 2, W / 2, 0.0]),
        np.array([-L / 2, -W / 2, 0.0]),
    ]
    hip_offsets_up = [np.array([o[0], o[1], h]) for o in hip_offsets]

    p_h_wd = [Twd2com[0:3, 3] + R @ o for o in hip_offsets]
    p_h_up = [Twd2com[0:3, 3] + R @ o for o in hip_offsets_up]

    chain1 = np.column_stack([p_h_wd[0], p_h_wd[1], p_h_wd[3], p_h_wd[2]])
    chain2 = np.column_stack([p_h_wd[0], p_h_wd[1], p_h_up[1], p_h_up[0]])
    chain3 = np.column_stack([p_h_wd[0], p_h_wd[2], p_h_up[2], p_h_up[0]])
    chain4 = np.column_stack([p_h_wd[2], p_h_wd[3], p_h_up[3], p_h_up[2]])
    chain5 = np.column_stack([p_h_wd[3], p_h_wd[1], p_h_up[1], p_h_up[3]])
    chain6 = np.column_stack([p_h_up[0], p_h_up[1], p_h_up[3], p_h_up[2]])

    # Body faces
    body_polys = [chain1.T, chain2.T, chain3.T, chain4.T, chain5.T, chain6.T]
    poly = Poly3DCollection(body_polys, facecolor=body_color, edgecolor=body_color, alpha=1.0)
    ax.add_collection3d(poly)

    # Legs
    for i_leg in range(4):
        _set_signs(p, i_leg)
        q = fcn_invKin3(Xt, pf34[:, i_leg], p)
        chain = _leg_kin(Twd2com, q, p)
        ax.plot(chain[0, :], chain[1, :], chain[2, :], color=leg_color, linewidth=3)

    # Feet markers
    ax.scatter(pf34[0, :], pf34[1, :], pf34[2, :], c=[leg_color], s=40, depthshade=False)

    # GRF arrows
    scale = 1e-2
    for i_leg in range(4):
        start = pf34[:, i_leg]
        end = pf34[:, i_leg] + scale * f34[:, i_leg]
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'r', linewidth=1.5)

    # External force
    p_ext_R = R @ p['p_ext'] + pcom.reshape(3)
    chain_Ue = np.column_stack([p_ext_R, p_ext_R + 0.01 * Ue.reshape(3)])
    ax.plot(chain_Ue[0, :], chain_Ue[1, :], chain_Ue[2, :], 'c', linewidth=1.5)

    # Ground
    if p['gait'] != -2:
        goffset = 0.5
        chain0 = np.array([
            [pcom[0, 0] - goffset, pcom[1, 0] + goffset, 0.0],
            [pcom[0, 0] + goffset, pcom[1, 0] + goffset, 0.0],
            [pcom[0, 0] + goffset, pcom[1, 0] - goffset, 0.0],
            [pcom[0, 0] - goffset, pcom[1, 0] - goffset, 0.0],
        ])
        ground_poly = Poly3DCollection([chain0], facecolor=ground_color, edgecolor=ground_color, alpha=0.6)
        ax.add_collection3d(ground_poly)


def fig_plot_robot_d(ax, Xd, Ud, p):
    """Plot the desired (ghost) robot pose with transparency."""
    L = p['L']
    W = p['W']
    h = p['h']
    body_color = p['body_color']
    leg_color = p['leg_color']

    pcom = Xd[0:3].reshape(3, 1)
    dpc = Xd[3:6].reshape(3, 1)
    R = Xd[6:15].reshape(3, 3, order='F')
    wb = Xd[15:18].reshape(3, 1)
    pf34 = Xd[18:30].reshape(3, 4, order='F')

    f34 = Ud.reshape(3, 4, order='F')

    Twd2com = np.block([
        [R, pcom],
        [np.zeros((1, 3)), np.array([[1.0]])],
    ])
    hip_offsets = [
        np.array([L / 2, W / 2, 0.0]),
        np.array([L / 2, -W / 2, 0.0]),
        np.array([-L / 2, W / 2, 0.0]),
        np.array([-L / 2, -W / 2, 0.0]),
    ]
    hip_offsets_up = [np.array([o[0], o[1], h]) for o in hip_offsets]

    p_h_wd = [Twd2com[0:3, 3] + R @ o for o in hip_offsets]
    p_h_up = [Twd2com[0:3, 3] + R @ o for o in hip_offsets_up]

    chain1 = np.column_stack([p_h_wd[0], p_h_wd[1], p_h_wd[3], p_h_wd[2]])
    chain2 = np.column_stack([p_h_wd[0], p_h_wd[1], p_h_up[1], p_h_up[0]])
    chain3 = np.column_stack([p_h_wd[0], p_h_wd[2], p_h_up[2], p_h_up[0]])
    chain4 = np.column_stack([p_h_wd[2], p_h_wd[3], p_h_up[3], p_h_up[2]])
    chain5 = np.column_stack([p_h_wd[3], p_h_wd[1], p_h_up[1], p_h_up[3]])
    chain6 = np.column_stack([p_h_up[0], p_h_up[1], p_h_up[3], p_h_up[2]])

    body_polys = [chain1.T, chain2.T, chain3.T, chain4.T, chain5.T, chain6.T]
    poly = Poly3DCollection(body_polys, facecolor=body_color, edgecolor=body_color, alpha=0.2)
    ax.add_collection3d(poly)

    scale = 1e-2
    for i_leg in range(4):
        start = pf34[:, i_leg]
        end = pf34[:, i_leg] + scale * f34[:, i_leg]
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'g', linewidth=1.5)


def fig_animate(tout, Xout, Uout, Xdout, Udout, Uext, p, save_path=None):
    """Animate the simulation results.

    If ``save_path`` is provided (e.g. ``test.mp4``), the animation is saved to
    that file. Otherwise the animation is shown interactively.
    """
    flag_movie = p['flag_movie']

    # Smoothen via interpolation on a uniform time grid
    t = np.arange(tout[0], tout[-1] + p['simTimeStep'], p['simTimeStep'])
    X = np.empty((t.size, Xout.shape[1]))
    U = np.empty((t.size, Uout.shape[1]))
    Xd = np.empty((t.size, Xdout.shape[1]))
    Ud = np.empty((t.size, Udout.shape[1]))
    Ue = np.empty((t.size, Uext.shape[1]))
    for j in range(Xout.shape[1]):
        X[:, j] = np.interp(t, tout, Xout[:, j])
    for j in range(Uout.shape[1]):
        U[:, j] = np.interp(t, tout, Uout[:, j])
    for j in range(Xdout.shape[1]):
        Xd[:, j] = np.interp(t, tout, Xdout[:, j])
    for j in range(Udout.shape[1]):
        Ud[:, j] = np.interp(t, tout, Udout[:, j])
    for j in range(Uext.shape[1]):
        Ue[:, j] = np.interp(t, tout, Uext[:, j])

    nt = t.size
    EA = np.zeros((nt, 3))
    EAd = np.zeros((nt, 3))
    for ii in range(nt):
        EA[ii, :] = fcn_X2EA(X[ii, :])
        EAd[ii, :] = fcn_X2EA(Xd[ii, :])

    # ZOH force plotting helper data
    t2 = np.repeat(t, 2)
    t2 = np.append(t2[1:], t2[-1])
    U2 = np.repeat(U, 2, axis=0)

    # Set up the figure
    fig = plt.figure(figsize=(12, 7), facecolor='white')

    ax_main = fig.add_subplot(3, 3, 1, projection='3d')
    ax_main2 = fig.add_subplot(3, 3, 2, projection='3d')
    # Combine the top-left 2x2: just use ax_main for the main 3D plot
    ax_x = fig.add_subplot(3, 3, 3)
    ax_dx = fig.add_subplot(3, 3, 6)
    ax_w = fig.add_subplot(3, 3, 9)
    ax_u = fig.add_subplot(3, 3, (7, 8))

    pcom0 = X[0, 0:3]

    def init():
        return []

    def update(ii):
        # Clear main 3D plot
        ax_main.cla()
        ax_main2.cla()

        pcom = X[ii, 0:3]
        ax_main.set_xlim(pcom[0] - 0.5, pcom[0] + 0.5)
        ax_main.set_ylim(pcom[1] - 0.5, pcom[1] + 0.5)
        ax_main.set_zlim(-0.2, 0.6)
        ax_main.set_box_aspect((1, 1, 1))
        ax_main.view_init(elev=15, azim=-60)

        fig_plot_robot(ax_main, X[ii, :], U[ii, :], Ue[ii, :], p)
        fig_plot_robot_d(ax_main, Xd[ii, :], np.zeros_like(Ud[ii, :]), p)

        # Annotations
        ax_main.text(pcom[0], pcom[1], 0.4, f't = {t[ii]:.2f}s')
        ax_main.text(pcom[0], pcom[1], 0.5, f'vd = {Xd[ii, 3]:.2f}m/s')
        ax_main.text(pcom[0], pcom[1], 0.45, f'v = {X[ii, 3]:.2f}m/s')

        # State subplots
        ax_x.cla()
        ax_x.plot(t[:ii + 1], X[:ii + 1, 0], 'r', t[:ii + 1], X[:ii + 1, 1], 'g', t[:ii + 1], X[:ii + 1, 2], 'b',
                  t[:ii + 1], Xd[:ii + 1, 0], 'r--', t[:ii + 1], Xd[:ii + 1, 1], 'g--', t[:ii + 1], Xd[:ii + 1, 2], 'b--', linewidth=1)
        ax_x.set_xlim(t[0], t[-1])
        ax_x.set_title('Position [m]')

        ax_dx.cla()
        ax_dx.plot(t[:ii + 1], X[:ii + 1, 3], 'r', t[:ii + 1], X[:ii + 1, 4], 'g', t[:ii + 1], X[:ii + 1, 5], 'b',
                   t[:ii + 1], Xd[:ii + 1, 3], 'r--', t[:ii + 1], Xd[:ii + 1, 4], 'g--', t[:ii + 1], Xd[:ii + 1, 5], 'b--', linewidth=1)
        ax_dx.set_xlim(t[0], t[-1])
        ax_dx.set_title('Velocity [m/s]')

        ax_w.cla()
        ax_w.plot(t[:ii + 1], X[:ii + 1, 15], 'r', t[:ii + 1], X[:ii + 1, 16], 'g', t[:ii + 1], X[:ii + 1, 17], 'b',
                  t[:ii + 1], Xd[:ii + 1, 15], 'r--', t[:ii + 1], Xd[:ii + 1, 16], 'g--', t[:ii + 1], Xd[:ii + 1, 17], 'b--', linewidth=1)
        ax_w.set_xlim(t[0], t[-1])
        ax_w.set_title('Angular velocity [rad/s]')

        ax_u.cla()
        ax_u.plot(t2[:2 * (ii + 1)], U2[:2 * (ii + 1), 2], 'r',
                  t2[:2 * (ii + 1)], U2[:2 * (ii + 1), 5], 'g',
                  t2[:2 * (ii + 1)], U2[:2 * (ii + 1), 8], 'b',
                  t2[:2 * (ii + 1)], U2[:2 * (ii + 1), 11], 'k',
                  t[:ii + 1], Ud[:ii + 1, 2], 'r--',
                  t[:ii + 1], Ud[:ii + 1, 5], 'g--',
                  t[:ii + 1], Ud[:ii + 1, 8], 'b--',
                  t[:ii + 1], Ud[:ii + 1, 11], 'k--', linewidth=1)
        ax_u.set_xlim(t[0], t[-1])
        ax_u.set_title('Fz [N]')

        fig.tight_layout()
        return []

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=range(0, nt, p['playSpeed']),
                                   interval=20, blit=False, repeat=False)

    if flag_movie and save_path is not None:
        anim.save(save_path, fps=30, writer='ffmpeg')
    else:
        plt.show()

    return t, EA, EAd
