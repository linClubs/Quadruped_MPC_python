"""Reference trajectory generators.

Python translation of fcns_MPC/fcn_bound_ref_traj.m and fcns/fcn_gen_XdUd.m.
"""

import os
import sys

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fcns.maps import hatMap
from fcns.bezier import bz_int


def fcn_bound_ref_traj(p):
    """Compute the periodic bounding reference trajectory and initial state.

    Mirrors MATLAB fcn_bound_ref_traj.m. Returns ``(p, Xt, Ut)`` where ``p`` is
    the (mutated) parameter dict carrying Bezier coefficients.
    """
    mass = p['mass']
    J = p['J']
    g = p['g']
    Tst = p['Tst']
    Tsw = p['Tsw']
    T = Tst + Tsw
    Tair = 0.5 * (Tsw - Tst)

    b_co = np.array([0.0, 0.8, 1.0, 1.0, 0.8, 0.0])
    b_ = b_co.mean()

    # Fz
    alpha_z = (mass * g * T) / (2.0 * b_ * Tst)
    Fz_co = alpha_z * b_co

    dz_co = bz_int(Fz_co / mass - g, 0.0, Tst)
    z_co = bz_int(dz_co, 0.0, Tst)

    # First-principle integration: initial vertical velocity
    dz0 = -1.0 / (Tst + Tair) * (
        z_co[-1] + Tair * (dz_co[-1] + g * Tst) - 0.5 * g * ((Tst + Tair) ** 2 - Tst ** 2)
    )

    dz_co = bz_int(Fz_co / mass - g, dz0, Tst)
    z_co = bz_int(dz_co, p['z0'], Tst)

    # theta (pitch)
    alpha_th = 140.0 * J[1, 1]
    tau_co = -alpha_th * b_co

    dth_co = bz_int(tau_co / J[1, 1], 0.0, Tst)
    dth0 = -0.5 * dth_co[-1]

    th0 = dth0 * Tair / 2.0
    dth_co = bz_int(tau_co / J[1, 1], dth0, Tst)
    th_co = bz_int(dth_co, th0, Tst)

    p['Fz_co'] = Fz_co
    p['dz_co'] = dz_co
    p['z_co'] = z_co
    p['tau_co'] = tau_co
    p['dth_co'] = dth_co
    p['th_co'] = th_co

    # Initial condition
    R0 = expm(hatMap(np.array([0.0, th0, 0.0])))
    Xt = np.zeros(30)
    Xt[0:3] = np.array([0.0, 0.0, p['z0']])
    Xt[3:6] = np.array([0.0, 0.0, dz0])
    Xt[6:15] = R0.flatten(order='F')
    Xt[15:18] = np.array([0.0, dth0, 0.0])
    Xt[18:30] = p['pf34'].flatten(order='F')

    Ut = np.tile(np.array([0.0, 0.0, 0.25 * mass * g]), (4, 1)).reshape(-1)
    return p, Xt, Ut


def fcn_gen_XdUd(t, Xt, bool_inStance, p):
    """Generate reference state and input trajectories for non-bound gaits.

    Mirrors MATLAB fcn_gen_XdUd.m.

    Parameters
    ----------
    t : (N,) array of sample times
    Xt : either an empty list (MATLAB []) or the current state vector. Used
        only to decide whether to apply the desired yaw; behaviour matches the
        MATLAB ``isempty(Xt)`` branch.
    bool_inStance : (4, N) boolean array of stance/swing flags per leg.
    """
    gait = p['gait']
    acc_d = p['acc_d']
    vel_d = p['vel_d']
    yaw_d = p['yaw_d']

    lent = len(t)
    Xd = np.zeros((30, lent))
    Ud = np.zeros((12, lent))
    Rground = p['Rground']

    for ii in range(lent):
        if gait >= 0:
            # Linear motion
            pc_d = np.array([0.0, 0.0, p['z0']])
            dpc_d = np.array([0.0, 0.0, 0.0])
            for jj in range(2):
                if t[ii] < (vel_d[jj] / acc_d):
                    dpc_d[jj] = acc_d * t[ii]
                    pc_d[jj] = 0.5 * acc_d * t[ii] ** 2
                else:
                    dpc_d[jj] = vel_d[jj]
                    pc_d[jj] = vel_d[jj] * t[ii] - 0.5 * vel_d[jj] ** 2 / acc_d
            # Angular motion
            if len(Xt) == 0:
                ea_d = np.array([0.0, 0.0, 0.0])
            else:
                ea_d = np.array([0.0, 0.0, yaw_d])
            vR_d = expm(hatMap(ea_d)).flatten(order='F')
            wb_d = np.zeros(3)
        pfd = (Rground @ p['pf34']).flatten(order='F')
        Xd[:, ii] = np.concatenate([pc_d, dpc_d, vR_d, wb_d, pfd])

        # Force distribution
        sum_inStance = int(np.sum(bool_inStance[:, ii]))
        if sum_inStance == 0:
            Ud[:, ii] = 0.0
        else:
            for k in range(4):
                if bool_inStance[k, ii]:
                    Ud[3 * k + 2, ii] = p['mass'] * p['g'] / sum_inStance
    return Xd, Ud
