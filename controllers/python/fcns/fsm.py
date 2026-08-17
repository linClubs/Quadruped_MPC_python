"""Finite State Machines for gait scheduling.

Python translation of fcns/fcn_FSM.m and fcns/fcn_FSM_bound.m.

Because MATLAB ``persistent`` variables are used to maintain FSM state across
calls, we encapsulate each FSM in a class. Each class is instantiated with the
initial state vector ``Xt`` and is callable thereafter with
``(t_, Xt, p)`` to produce ``(FSMout, Xd, Ud, Xt)``.
"""

import os
import sys

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fcns.maps import hatMap
from fcns.bezier import polyval_bz
from fcns_mpc.ref_traj import fcn_gen_XdUd, fcn_bound_ref_traj


def _clamp_s(s):
    s = np.asarray(s, dtype=float)
    s = np.where(s < 0.0, 0.0, s)
    s = np.where(s > 1.0, 1.0, s)
    return s


class FSMSimple:
    """General FSM for trot/pacing/gallop/crawl/pose/backflip.

    Python translation of fcn_FSM.m.
    """

    def __init__(self, Xt, p):
        self.FSM = np.zeros(4)
        self.Ta = np.zeros(4)
        self.Tb = np.ones(4)
        self.pf_R_trans = Xt[18:30].copy()
        self._initialized = False
        self.p = p

    def __call__(self, t_, Xt, p):
        L = p['L']
        W = p['W']
        d = p['d']
        gait = p['gait']
        Tst_ = p['Tst']
        # Tst = min(Tst_, 0.2 / max(np.linalg.norm(Xt[3:5]), 1e-12))
        Tst = min(Tst_, 0.2 / max(np.linalg.norm(Xt[3:5]), 1e-9))
        Tsw = p['Tsw']
        T = Tst + Tsw
        Tair = 0.5 * (Tsw - Tst)

        pc = Xt[0:3].copy()
        dpc = Xt[3:6].copy()
        vR = Xt[6:15].copy()
        wb = Xt[15:18].copy()
        R = vR.reshape(3, 3, order='F')
        idx_pf = slice(18, 30)
        pf34 = Xt[idx_pf].reshape(3, 4, order='F')

        FSM = self.FSM
        Ta = self.Ta
        Tb = self.Tb
        pf_R_trans = self.pf_R_trans

        t = t_[0]
        s = np.zeros(4)

        if not self._initialized:
            # Initialize Ta/Tb for the first call, depending on gait
            for i_leg in range(4):
                if gait == -3:
                    Ta[i_leg] = t
                    Tb[0:2] = Ta[0:2] + p['Tds'] * np.ones(2) if 'Tds' in p else Ta[0:2]
                    Tb[2:4] = Ta[2:4] + p['Tbs'] * np.ones(2) if 'Tbs' in p else Ta[2:4]
                elif gait == -1:
                    Ta[i_leg] = 0.0
                    Tb[i_leg] = -1.0
                elif gait == 1:
                    Ta[0:2] = np.array([t, t])
                    Ta[2:4] = np.array([t, t]) + 0.5 * (Tst + Tsw)
                    Tb[0:2] = Ta[0:2] + Tst
                    Tb[2:4] = Ta[2:4] + Tst + Tair
                elif gait == 2:
                    Ta[0:3:2] = np.array([t, t])
                    Ta[1:4:2] = np.array([t, t]) + 0.5 * (Tst + Tsw)
                    Tb[i_leg] = Ta[i_leg] + Tst
                elif gait == 3:
                    Ta[0] = t
                    Ta[1] = t + 0.05
                    Ta[2] = t + 0.05 + Tst
                    Ta[3] = t + 0.1 + Tst
                    Tb[i_leg] = Ta[i_leg] + Tst
                elif gait == 5:
                    Ta[0] = t
                    Ta[1] = t + Tsw
                    Ta[2] = t + Tsw * 2
                    Ta[3] = t + Tsw * 3
                    Tb[i_leg] = Ta[i_leg] + Tst
                else:
                    Ta[0:4:3] = np.array([t, t])
                    Ta[1:3] = np.array([t, t]) + 0.5 * (Tst + Tsw)
                    Tb[i_leg] = Ta[i_leg] + Tst
                FSM[i_leg] = FSM[i_leg] + 1
            pf_R_trans = Xt[idx_pf].copy()
            self._initialized = True

        # Per-leg FSM transitions
        for i_leg in range(4):
            s[i_leg] = (t - Ta[i_leg]) / (Tb[i_leg] - Ta[i_leg]) if (Tb[i_leg] - Ta[i_leg]) != 0 else 0.0
        s = _clamp_s(s)
        # Recompute transitions on this step (mirror the per-leg logic in MATLAB)
        for i_leg in range(4):
            if FSM[i_leg] == 0:
                # Already handled in initialization for the first call only
                pass
            elif FSM[i_leg] == 1 and s[i_leg] >= 1 - 1e-7:
                FSM[i_leg] = FSM[i_leg] + 1
                Ta[i_leg] = t
                Tb[i_leg] = Ta[i_leg] + Tsw
                pf_R_trans = Xt[idx_pf].copy()
            elif FSM[i_leg] == 2 and s[i_leg] >= 1 - 1e-7:
                if gait == -3:
                    ph = R @ np.array([[L / 2, L / 2], [W / 2 + d, -W / 2 - d], [0, 0]]) + np.tile(pc.reshape(3, 1), (1, 2))
                    pf = ph + np.tile(np.array([[0], [0], [-0.18]]), (1, 2))
                    if pf[2, 0] < 1e-4:
                        FSM[i_leg] = 1
                        Ta[i_leg] = t
                        Tb[i_leg] = Ta[i_leg] + Tst
                        pf_R_trans = Xt[idx_pf].copy()
                else:
                    FSM[i_leg] = 1
                    Ta[i_leg] = t
                    Tb[i_leg] = Ta[i_leg] + Tst
                    pf_R_trans = Xt[idx_pf].copy()
        s = (t - Ta) / np.where((Tb - Ta) == 0, 1.0, (Tb - Ta))
        s = _clamp_s(s)

        # FSM over prediction horizon
        FSM_ = np.tile(FSM.reshape(4, 1), (1, p['predHorizon']))
        for i_leg in range(4):
            for ii in range(1, p['predHorizon']):
                if t_[ii] <= Ta[i_leg]:
                    FSM_[i_leg, ii] = 1
                elif (Ta[i_leg] < t_[ii]) and (t_[ii] < Tb[i_leg]):
                    FSM_[i_leg, ii] = FSM[i_leg]
                elif Ta[i_leg] + Tst + Tsw < t_[ii]:
                    FSM_[i_leg, ii] = FSM[i_leg]
                else:
                    if FSM[i_leg] == 1:
                        FSM_[i_leg, ii] = 2
                    else:
                        FSM_[i_leg, ii] = 1
        if gait == -1:
            FSM_ = np.ones_like(FSM_)
        elif gait == -3:
            FSM[2:4] = 1
            FSM_[:, 2:4] = 1

        bool_inStance = (FSM_ == 1)

        # Generate reference trajectory
        Xd, Ud = fcn_gen_XdUd(t_, Xt, bool_inStance, p)
        p = fcn_bound_ref_traj(p)[0]

        if gait == 1:
            for ii in range(p['predHorizon']):
                fsm = FSM_[:, ii]
                if fsm[0] == 1:
                    s_ph = (t_[ii] - Ta[0]) / (Tb[0] - Ta[0]) if (Tb[0] - Ta[0]) != 0 else 0.0
                    s_ph = max(0.0, min(1.0, s_ph))

                    th_d = polyval_bz(-p['th_co'], s_ph)
                    dth_d = polyval_bz(-p['dth_co'], s_ph)
                    z_d = polyval_bz(p['z_co'], s_ph)
                    vR_d = expm(hatMap(np.array([0.0, th_d, 0.0]))).flatten(order='F')
                    Xd[2, ii] = z_d
                    Xd[6:15, ii] = vR_d
                    Xd[16, ii] = dth_d

                    Fz_d = polyval_bz(p['Fz_co'], s_ph)
                    tau_d = polyval_bz(p['tau_co'], s_ph)
                    r = pf34[:, 0] - pc
                    Ud[2, ii] = 0.5 * Fz_d
                    Ud[5, ii] = 0.5 * Fz_d
                    Ud[0, ii] = 0.5 * (r[0] * Fz_d - tau_d) / r[2]
                    Ud[3, ii] = 0.5 * (r[0] * Fz_d - tau_d) / r[2]
                elif fsm[2] == 1:
                    s_ph = (t_[ii] - Ta[2]) / (Tb[2] - Ta[2]) if (Tb[2] - Ta[2]) != 0 else 0.0
                    s_ph = max(0.0, min(1.0, s_ph))

                    th_d = polyval_bz(-p['th_co'], s_ph)
                    dth_d = polyval_bz(-p['dth_co'], s_ph)
                    z_d = polyval_bz(p['z_co'], s_ph)
                    vR_d = expm(hatMap(np.array([0.0, th_d, 0.0]))).flatten(order='F')
                    Xd[2, ii] = z_d
                    Xd[6:15, ii] = vR_d
                    Xd[16, ii] = dth_d

                    Fz_d = polyval_bz(p['Fz_co'], s_ph)
                    tau_d = polyval_bz(-p['tau_co'], s_ph)
                    r = pf34[:, 2] - pc
                    Ud[8, ii] = 0.5 * Fz_d
                    Ud[11, ii] = 0.5 * Fz_d
                    Ud[6, ii] = 0.5 * (r[0] * Fz_d - tau_d) / r[2]
                    Ud[9, ii] = 0.5 * (r[0] * Fz_d - tau_d) / r[2]

        # Swing-leg trajectory (Raibert-style capture-point placement)
        p_hip_b = np.array([
            [L / 2, L / 2, -L / 2, -L / 2],
            [W / 2 + d, -W / 2 - d, W / 2 + d, -W / 2 - d],
            [0.0, 0.0, 0.0, 0.0],
        ])
        p_hip_R = R @ p_hip_b
        ws = R @ wb
        v_hip_R = np.tile(dpc.reshape(3, 1), (1, 4)) + hatMap(ws) @ p_hip_R

        p_cap = np.zeros((2, 4))
        vd = Xd[3:5, 0]
        for i_leg in range(4):
            temp = 0.8 * Tst * vd + np.sqrt(p['z0'] / p['g']) * (v_hip_R[0:2, i_leg] - vd)
            temp = np.where(temp < -0.15, -0.15, temp)
            temp = np.where(temp > 0.15, 0.15, temp)
            p_cap[:, i_leg] = pc[0:2] + p_hip_R[0:2, i_leg] + temp

        if p['gait'] == -3:
            pfd = Xt[idx_pf].copy()
            for i_leg in range(2):
                if FSM[i_leg] == 2:
                    ph = R @ np.array([[L / 2, L / 2], [W / 2 + d, -W / 2 - d], [0, 0]]) + np.tile(pc.reshape(3, 1), (1, 2))
                    pf = ph + np.tile(np.array([[0], [0], [-0.18]]), (1, 2))
                    pf[2, :] = np.maximum(0.0, pf[2, :])
                    pfd[0:6] = pf.flatten(order='F')
            Xd[idx_pf, :] = np.tile(pfd.reshape(-1, 1), (1, p['predHorizon']))
        elif p['gait'] == -2:
            raise NotImplementedError("GOT mode requires fcn_GOT_Rg_BB which is not ported.")
        else:
            pfd = Xt[idx_pf].copy()
            for i_leg in range(4):
                idx = slice(3 * i_leg, 3 * i_leg + 3)
                if FSM[i_leg] == 2:
                    co_x = np.linspace(pf_R_trans[idx][0], p_cap[0, i_leg], 6)
                    co_y = np.linspace(pf_R_trans[idx][1], p_cap[1, i_leg], 6)
                    co_z = np.array([0.0, 0.0, 0.1, 0.1, 0.0, -0.002])
                    pfd[idx] = np.array([
                        polyval_bz(co_x, s[i_leg]),
                        polyval_bz(co_y, s[i_leg]),
                        polyval_bz(co_z, s[i_leg]),
                    ])
            Xd[idx_pf, :] = np.tile(pfd.reshape(-1, 1), (1, p['predHorizon']))

        # Persist state for next call
        self.FSM = FSM
        self.Ta = Ta
        self.Tb = Tb
        self.pf_R_trans = pf_R_trans

        return FSM, Xd, Ud, Xt


class FSMBound:
    """Bounding-gait FSM.

    Python translation of fcn_FSM_bound.m.
    """

    def __init__(self, Xt, p):
        self.FSM = 1
        self.Ta = 0.0
        self.Tb = None
        self.pf_trans = Xt[18:30].copy()
        self.Ta_sw = None
        self.Tb_sw = None
        self._initialized = False
        self.p = p

    def __call__(self, t_, Xt, p):
        acc = 1.0
        vd = np.array([2.0, 0.0])

        Tst_ = p['Tst']
        # Tst = min(Tst_, 0.2 / max(np.linalg.norm(Xt[3:5]), 1e-12))
        Tst = min(Tst_, 0.2 / max(np.linalg.norm(Xt[3:5]), 1e-9))
        Tsw = p['Tsw']
        T = Tst + Tsw
        Tair = 0.5 * (Tsw - Tst)

        # Update p with periodic bounding coefficients
        p = fcn_bound_ref_traj(p)[0]

        pc = Xt[0:3].copy()
        dpc = Xt[3:6].copy()
        vR = Xt[6:15].copy()
        wb = Xt[15:18].copy()
        R = vR.reshape(3, 3, order='F')
        idx_pf = slice(18, 30)
        pf = Xt[idx_pf].copy()
        pf34 = pf.reshape(3, 4, order='F')

        FSM = self.FSM
        Ta = self.Ta
        Tb = self.Tb
        pf_trans = self.pf_trans
        Ta_sw = self.Ta_sw
        Tb_sw = self.Tb_sw

        if not self._initialized:
            Ta = 0.0
            Tb = Ta + Tst
            pf_trans = pf.copy()
            Ta_sw = np.zeros(4)
            Tb_sw = np.zeros(4)
            Ta_sw[0:2] = Tst - T
            Tb_sw[0:2] = Ta_sw[0:2] + Tsw
            Ta_sw[2:4] = -Tair
            Tb_sw[2:4] = Ta_sw[2:4] + Tsw
            self._initialized = True

        # FSM transitions
        t = t_[0]
        s = (t - Ta) / (Tb - Ta) if (Tb - Ta) != 0 else 0.0
        if FSM == 1 and s >= 1:
            Ta = Tb
            Tb = Tb + Tair
            FSM = FSM + 1
            Ta_sw[0:2] = Ta_sw[0:2] + T
            Tb_sw[0:2] = Ta_sw[0:2] + Tsw
            pf_trans[0:6] = pf[0:6]
        elif FSM == 2 and s >= 1:
            Ta = Tb
            Tb = Tb + Tst
            FSM = FSM + 1
        elif FSM == 3 and s >= 1:
            Ta = Tb
            Tb = Tb + Tair
            FSM = FSM + 1
            Ta_sw[2:4] = Ta_sw[2:4] + T
            Tb_sw[2:4] = Ta_sw[2:4] + Tsw
            pf_trans[6:12] = pf[6:12]
        elif FSM == 4 and s >= 1:
            Ta = Tb
            Tb = Tb + Tst
            FSM = 1

        s = (t - Ta) / (Tb - Ta) if (Tb - Ta) != 0 else 0.0
        s = max(0.0, min(1.0, s))

        FSM_ = np.zeros(p['predHorizon'])
        s_ = np.zeros(p['predHorizon'])
        Ud = np.zeros((12, p['predHorizon']))
        Xd = np.tile(Xt.reshape(-1, 1), (1, p['predHorizon']))

        pd = np.array([0.0, 0.0])
        dpd = np.array([0.0, 0.0])

        for ii in range(p['predHorizon']):
            FSM_ii, s_ii = self._fsm_pred_hor(FSM, Ta, t_[ii], p, Tst, Tsw, T, Tair)
            FSM_[ii] = FSM_ii
            s_[ii] = s_ii
            FSM_[0] = FSM
            s_[0] = s

            for dir_xy in range(2):
                if t_[ii] < (vd[dir_xy] / acc):
                    dpd[dir_xy] = acc * t_[ii]
                    pd[dir_xy] = 0.5 * acc * t_[ii] ** 2
                else:
                    dpd[dir_xy] = vd[dir_xy]
                    pd[dir_xy] = vd[dir_xy] * t_[ii] - 0.5 * vd[dir_xy] ** 2 / acc

            if FSM_[ii] == 1:
                Fz_d = polyval_bz(p['Fz_co'], s_[ii])
                dz_d = polyval_bz(p['dz_co'], s_[ii])
                z_d = polyval_bz(p['z_co'], s_[ii])

                tau_d = polyval_bz(p['tau_co'], s_[ii])
                dth_d = polyval_bz(p['dth_co'], s_[ii])
                th_d = polyval_bz(p['th_co'], s_[ii])
                R_d = expm(hatMap(np.array([0.0, th_d, 0.0])))

                dx = pc[0] - pf34[0, 0]
                Fx_d = (dx * Fz_d - tau_d) / (z_d)

                Ud[0:6, ii] = 0.5 * np.array([Fx_d, 0.0, Fz_d, Fx_d, 0.0, Fz_d])
                Xd[0:18, ii] = np.concatenate([
                    pd, np.array([z_d]),
                    dpd, np.array([dz_d]),
                    R_d.flatten(order='F'),
                    np.array([0.0, dth_d, 0.0]),
                ])
            elif FSM_[ii] == 2:
                dz_d = p['dz_co'][-1] - p['g'] * (s_[ii] * Tair)
                z_d = p['z_co'][-1] + p['dz_co'][-1] * (s_[ii] * Tair) - 0.5 * p['g'] * (s_[ii] * Tair) ** 2
                dth_d = p['dth_co'][-1]
                th_d = p['th_co'][-1] + dth_d * (s_[ii] * Tair)
                R_d = expm(hatMap(np.array([0.0, th_d, 0.0])))
                Xd[0:18, ii] = np.concatenate([
                    pd, np.array([z_d]),
                    dpd, np.array([dz_d]),
                    R_d.flatten(order='F'),
                    np.array([0.0, dth_d, 0.0]),
                ])
            elif FSM_[ii] == 3:
                Fz_d = polyval_bz(p['Fz_co'], s_[ii])
                dz_d = polyval_bz(p['dz_co'], s_[ii])
                z_d = polyval_bz(p['z_co'], s_[ii])

                tau_d = polyval_bz(-p['tau_co'], s_[ii])
                dth_d = polyval_bz(-p['dth_co'], s_[ii])
                th_d = polyval_bz(-p['th_co'], s_[ii])
                R_d = expm(hatMap(np.array([0.0, th_d, 0.0])))

                dx = pc[0] - pf34[0, 2]
                Fx_d = (dx * Fz_d - tau_d) / (z_d)

                Ud[6:12, ii] = 0.5 * np.array([Fx_d, 0.0, Fz_d, Fx_d, 0.0, Fz_d])
                Xd[0:18, ii] = np.concatenate([
                    pd, np.array([z_d]),
                    dpd, np.array([dz_d]),
                    R_d.flatten(order='F'),
                    np.array([0.0, dth_d, 0.0]),
                ])
            elif FSM_[ii] == 4:
                dz_d = p['dz_co'][-1] - p['g'] * (s_[ii] * Tair)
                z_d = p['z_co'][-1] + p['dz_co'][-1] * (s_[ii] * Tair) - 0.5 * p['g'] * (s_[ii] * Tair) ** 2
                dth_d = -p['dth_co'][-1]
                th_d = -p['th_co'][-1] + dth_d * (s_[ii] * Tair)
                R_d = expm(hatMap(np.array([0.0, th_d, 0.0])))
                Xd[0:18, ii] = np.concatenate([
                    pd, np.array([z_d]),
                    dpd, np.array([dz_d]),
                    R_d.flatten(order='F'),
                    np.array([0.0, dth_d, 0.0]),
                ])

        # Swing-leg trajectory
        L = p['L']
        W = p['W']
        d = p['d']
        p_hip_b = np.array([
            [L / 2, L / 2, -L / 2, -L / 2],
            [W / 2 + d, -W / 2 - d, W / 2 + d, -W / 2 - d],
            [0.0, 0.0, 0.0, 0.0],
        ])
        p_hip_R = R @ p_hip_b
        ws = R @ wb
        v_hip_R = np.tile(dpc.reshape(3, 1), (1, 4)) + hatMap(ws) @ p_hip_R

        p_cap = np.zeros((2, 4))
        dpd0 = Xd[3:5, 0]
        for i_leg in range(4):
            temp = 0.8 * Tst * dpd0 + np.sqrt(p['z0'] / p['g']) * (v_hip_R[0:2, i_leg] - dpd0)
            temp = np.where(temp < -0.15, -0.15, temp)
            temp = np.where(temp > 0.15, 0.15, temp)
            p_cap[:, i_leg] = pc[0:2] + p_hip_R[0:2, i_leg] + temp

        pfd = Xt[idx_pf].copy()
        s_sw = (t - Ta_sw) / np.where((Tb_sw - Ta_sw) == 0, 1.0, (Tb_sw - Ta_sw))
        s_sw = np.where(s_sw < 0.0, 0.0, s_sw)
        s_sw = np.where(s_sw > 1.0, 1.0, s_sw)

        def _fill_swing(legs):
            for i_leg in legs:
                idx = slice(3 * i_leg, 3 * i_leg + 3)
                co_x = np.linspace(pf_trans[idx][0], p_cap[0, i_leg], 6)
                co_y = np.linspace(pf_trans[idx][1], p_cap[1, i_leg], 6)
                co_z = np.array([0.0, 0.0, 0.15, 0.15, 0.0, -0.002])
                pfd[idx] = np.array([
                    polyval_bz(co_x, s_sw[i_leg]),
                    polyval_bz(co_y, s_sw[i_leg]),
                    polyval_bz(co_z, s_sw[i_leg]),
                ])

        if FSM == 1:
            _fill_swing([2, 3])
        elif FSM == 2:
            _fill_swing([0, 1, 2, 3])
        elif FSM == 3:
            _fill_swing([0, 1])
        elif FSM == 4:
            _fill_swing([0, 1, 2, 3])

        Xd[idx_pf, :] = np.tile(pfd.reshape(-1, 1), (1, p['predHorizon']))
        Xd = np.tile(Xd[:, 0].reshape(-1, 1), (1, p['predHorizon']))

        # Persist state for next call
        self.FSM = FSM
        self.Ta = Ta
        self.Tb = Tb
        self.pf_trans = pf_trans
        self.Ta_sw = Ta_sw
        self.Tb_sw = Tb_sw

        return FSM, Xd, Ud, Xt

    @staticmethod
    def _fsm_pred_hor(FSM, Ta, t, p, Tst, Tsw, T, Tair):
        Tst = p['Tst']
        Tsw = p['Tsw']
        T = Tst + Tsw
        Tair = 0.5 * (Tsw - Tst)

        tp = np.mod(t - Ta, T)

        if FSM == 1 or FSM == 3:
            Tnode = np.array([Tst, Tair, Tst, Tair])
        else:
            Tnode = np.array([Tair, Tst, Tair, Tst])

        for ii in range(4):
            if tp <= np.sum(Tnode[0:ii + 1]):
                FSMout = int(np.mod(FSM + ii - 1, 4))
                if FSMout == 0:
                    FSMout = 4
                if ii == 0:
                    sout = tp / Tnode[ii]
                else:
                    sout = (tp - np.sum(Tnode[0:ii])) / Tnode[ii]
                return FSMout, sout
        # Fallback (should not be reached)
        return FSM, 1.0
