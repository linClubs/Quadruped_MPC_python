"""Model Predictive Control utilities.

Python translation of:
    fcns_MPC/fcn_get_ABD_eta.m
    fcns_MPC/fcn_get_QP_form_eta.m
    fcns_MPC/fcn_get_disturbance.m
"""

import os
import sys

import numpy as np
from scipy.linalg import logm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fcns.maps import hatMap, veeMap
from fcns.bezier import polyval_bz


def _fcn_get_N():
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


def _fcn_get_D(in_):
    d = in_[0]
    e = in_[1]
    f = in_[2]
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


def _fcn_get_F(k):
    k = k.reshape(3)
    return np.array([
        [k[0], 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, k[1], 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, k[2], 0, 0],
    ], dtype=float)


def _eta_co_xv(fop, dt, mass, g):
    Cx_x = np.eye(3)
    Cx_v = np.eye(3) * dt
    Cv_v = np.eye(3)
    Cv_u = dt / mass * np.hstack([np.eye(3)] * 4)
    Cv_c = Cv_u @ fop + np.array([0.0, 0.0, -g]) * dt
    return Cx_x, Cx_v, Cv_v, Cv_u, Cv_c


def _eta_co_R(Rop, wop, dt):
    N = _fcn_get_N()
    invN = np.linalg.pinv(N)

    C_eta = np.kron(np.eye(3), Rop @ hatMap(wop)) @ N + np.kron(np.eye(3), Rop) @ _fcn_get_D(wop)
    C_w = np.kron(np.eye(3), Rop) @ N
    C_c = (Rop @ hatMap(wop)).flatten(order='F').reshape(-1, 1) - np.kron(np.eye(3), Rop) @ N @ wop.reshape(-1, 1)

    CE_eta = np.eye(3) + dt * (invN @ np.kron(np.eye(3), Rop.T) @ C_eta)
    CE_w = dt * (invN @ np.kron(np.eye(3), Rop.T) @ C_w)
    CE_c = (dt * (invN @ np.kron(np.eye(3), Rop.T) @ C_c)).reshape(-1)
    return CE_eta, CE_w, CE_c


def _eta_co_w(xop, Rop, wop, fop, dt, J, pf):
    N = _fcn_get_N()
    r1 = pf[:, 0] - xop
    r2 = pf[:, 1] - xop
    r3 = pf[:, 2] - xop
    r4 = pf[:, 3] - xop
    Mop = np.hstack([hatMap(r1), hatMap(r2), hatMap(r3), hatMap(r4)]) @ fop

    temp_J_w = hatMap(J @ wop) - hatMap(wop) @ J
    sum_fop = np.hstack([np.eye(3)] * 4) @ fop

    Cx = Rop.T @ hatMap(sum_fop)
    Ceta = _fcn_get_F((Rop.T @ Mop).reshape(-1, 1)) @ N - temp_J_w @ hatMap(wop)
    Cw = temp_J_w
    Cu = Rop.T @ np.hstack([hatMap(r1), hatMap(r2), hatMap(r3), hatMap(r4)])
    Cc = -hatMap(wop) @ J @ wop + Rop.T @ Mop - temp_J_w @ wop - Cx @ xop

    Cw_x = dt * np.linalg.solve(J, Cx)
    Cw_eta = dt * np.linalg.solve(J, Ceta)
    Cw_w = dt * np.linalg.solve(J, Cw) + np.eye(3)
    Cw_u = dt * np.linalg.solve(J, Cu)
    Cw_c = dt * np.linalg.solve(J, Cc)
    return Cw_x, Cw_eta, Cw_w, Cw_u, Cw_c


def fcn_get_ABD_eta(Xt, Ut, p):
    """Build linearized dynamics matrices (A, B, D) for the eta-MPC."""
    dt = p['Tmpc']

    xop = Xt[0:3].reshape(3, 1)
    vop = Xt[3:6].reshape(3, 1)
    Rop = Xt[6:15].reshape(3, 3, order='F')
    wop = Xt[15:18].reshape(3, 1)
    pf34 = Xt[18:30].reshape(3, 4, order='F')

    Ut = Ut.reshape(-1)

    Cx_x, Cx_v, Cv_v, Cv_u, Cv_c = _eta_co_xv(Ut, dt, p['mass'], p['g'])
    CE_eta, CE_w, CE_c = _eta_co_R(Rop, wop, dt)
    Cw_x, Cw_eta, Cw_w, Cw_u, Cw_c = _eta_co_w(xop.reshape(3), Rop, wop.reshape(3), Ut.reshape(-1), dt, p['J'], pf34)

    A = np.block([
        [Cx_x, Cx_v, np.zeros((3, 6))],
        [np.zeros((3, 3)), Cv_v, np.zeros((3, 6))],
        [np.zeros((3, 6)), CE_eta, CE_w],
        [Cw_x, np.zeros((3, 3)), Cw_eta, Cw_w],
    ])
    B = np.vstack([
        np.zeros((3, 12)),
        Cv_u,
        np.zeros((3, 12)),
        Cw_u,
    ])
    D = np.concatenate([
        np.zeros(3),
        Cv_c.reshape(-1),
        CE_c.reshape(-1),
        Cw_c.reshape(-1),
    ])
    return A, B, D


def fcn_get_QP_form_eta(Xt, Ut, Xd, Ud, p):
    """Build the QP matrices for the eta-MPC.

    Mirrors MATLAB fcn_get_QP_form_eta.m. Returns H, g, Aineq, bineq, Aeq, beq
    so that the QP is

        min 0.5 z' H z + g' z
        s.t. Aineq z <= bineq, Aeq z = beq
    """
    mu = p['mu']
    n_hor = p['predHorizon']
    Umax = p['Umax']
    decayRate = p['decayRate']

    R_cost = p['R']
    Q = p['Q']
    Qf = p['Qf']
    Qx = Q[0:3, 0:3]
    Qv = Q[3:6, 3:6]
    Qeta = Q[6:9, 6:9]
    Qw = Q[9:12, 9:12]
    Qxf = Qf[0:3, 0:3]
    Qvf = Qf[3:6, 3:6]
    Qetaf = Qf[6:9, 6:9]
    Qwf = Qf[9:12, 9:12]

    nX = 12
    nU = 12

    A, B, d = fcn_get_ABD_eta(Xt, Ut, p)

    Rt = Xt[6:15].reshape(3, 3, order='F')
    qt = np.concatenate([Xt[0:6], np.zeros(3), Xt[15:18]])

    # Fz bounds
    Fzd = Ud[[2, 5, 8, 11], :]
    lb = -1.0 * Fzd
    ub = 2.0 * Fzd

    H = np.zeros(((nX + nU) * n_hor, (nX + nU) * n_hor))
    g = np.zeros((nX + nU) * n_hor)
    Aeq = np.zeros((nX * n_hor, (nX + nU) * n_hor))
    beq = np.zeros(nX * n_hor)
    if p['gait'] == -2:
        Aineq_unit = np.array([
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ], dtype=float)
    else:
        Aineq_unit = np.array([
            [1, 0, -mu],
            [-1, 0, -mu],
            [0, 1, -mu],
            [0, -1, -mu],
            [0, 0, 1],
            [0, 0, -1],
        ], dtype=float)
    nAineq_unit = Aineq_unit.shape[0]
    Aineq = np.zeros((4 * nAineq_unit * n_hor, (nX + nU) * n_hor))
    bineq = np.zeros(4 * nAineq_unit * n_hor)

    Ut = Ut.reshape(-1)

    for i_hor in range(n_hor):
        xd = Xd[0:3, i_hor]
        vd = Xd[3:6, i_hor]
        Rd = Xd[6:15, i_hor].reshape(3, 3, order='F')
        wd = Xd[15:18, i_hor]

        idx_u = slice(i_hor * (nX + nU), i_hor * (nX + nU) + nU)
        idx_x = slice(i_hor * (nX + nU) + nU, i_hor * (nX + nU) + nU + nX)

        if i_hor == n_hor - 1:
            H[idx_x, idx_x] = Qf * decayRate ** i_hor
            g[idx_x] = np.concatenate([
                -Qxf @ xd,
                -Qvf @ vd,
                Qetaf @ veeMap(logm(Rd.T @ Rt)),
                -Qwf @ wd,
            ]) * decayRate ** i_hor
        else:
            H[idx_x, idx_x] = Q * decayRate ** i_hor
            g[idx_x] = np.concatenate([
                -Qx @ xd,
                -Qv @ vd,
                Qeta @ veeMap(logm(Rd.T @ Rt)),
                -Qw @ wd,
            ]) * decayRate ** i_hor
        H[idx_u, idx_u] = R_cost * decayRate ** i_hor
        g[idx_u] = R_cost.T @ (Ut - Ud[:, i_hor]) * decayRate ** i_hor

        # Equality constraints (dynamics)
        if i_hor == 0:
            Aeq[0:nX, 0:(nU + nX)] = np.hstack([-B, np.eye(nX)])
            beq[0:nX] = A @ qt + d
        else:
            row_slice = slice(i_hor * nX, i_hor * nX + nX)
            col_start = (i_hor - 1) * (nX + nU) + nU
            Aeq[row_slice, col_start:col_start + (2 * nX + nU)] = np.hstack([-A, -B, np.eye(nX)])
            beq[row_slice] = d

        # Inequality constraints (friction cone + Fz bounds)
        Fi = np.zeros((4 * nAineq_unit, 12))
        hi = np.zeros(4 * nAineq_unit)
        for i_leg in range(4):
            idx_F = slice(i_leg * nAineq_unit, i_leg * nAineq_unit + nAineq_unit)
            idx_u_leg = slice(i_leg * 3, i_leg * 3 + 3)
            Fi[idx_F, idx_u_leg] = Aineq_unit
            if p['gait'] == -2:
                hi[idx_F] = np.array([
                    Umax - Ut[idx_u_leg.start + 0],
                    Umax + Ut[idx_u_leg.start + 0],
                    Umax - Ut[idx_u_leg.start + 1],
                    Umax + Ut[idx_u_leg.start + 1],
                    Umax - Ut[idx_u_leg.start + 2],
                    Umax + Ut[idx_u_leg.start + 2],
                ])
            else:
                hi[idx_F] = np.array([
                    mu * Ut[idx_u_leg.start + 2] - Ut[idx_u_leg.start + 0],
                    mu * Ut[idx_u_leg.start + 2] + Ut[idx_u_leg.start + 0],
                    mu * Ut[idx_u_leg.start + 2] - Ut[idx_u_leg.start + 1],
                    mu * Ut[idx_u_leg.start + 2] + Ut[idx_u_leg.start + 1],
                    ub[i_leg, i_hor] - Ut[idx_u_leg.start + 2] + Ud[idx_u_leg.start + 2, i_hor],
                    -lb[i_leg, i_hor] + Ut[idx_u_leg.start + 2] - Ud[idx_u_leg.start + 2, i_hor],
                ])
        idx_A = slice(i_hor * 4 * nAineq_unit, i_hor * 4 * nAineq_unit + 4 * nAineq_unit)
        idx_z = slice(i_hor * (nX + nU), i_hor * (nX + nU) + nU)
        Aineq[idx_A, idx_z] = Fi
        bineq[idx_A] = hi

    return H, g, Aineq, bineq, Aeq, beq


def fcn_get_disturbance(t, p):
    """External disturbance force and application point (body frame)."""
    bz_w = np.array([0.0, 0.5, 1.0, 1.0, 0.5, 0.0])
    if 0.5 <= t <= 1.3:
        s_w = (t - 0.5) / 0.8
        w = polyval_bz(8.0 * bz_w, s_w)
    elif 2.3 <= t <= 3.1:
        s_w = (t - 2.3) / 0.8
        w = polyval_bz(22.0 * bz_w, s_w)
    else:
        w = 0.0

    u_ext = np.array([0.0, w, 0.0])
    p_ext = np.array([p['L'] / 2.0, p['W'] / 2.0, p['d']])
    return u_ext, p_ext
