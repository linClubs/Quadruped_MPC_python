"""Single rigid body (SRB) dynamics.

Python translation of dynamics_SRB.m.

The state vector X has 30 entries:
    X[0:3]   = pc     (COM position in world frame)
    X[3:6]   = dpc    (COM linear velocity)
    X[6:15]  = R      (rotation matrix, column-major flatten)
    X[15:18] = wb     (body angular velocity)
    X[18:30] = pf34   (4 foot positions in world frame, column-major)
"""

import numpy as np

from .maps import hatMap


def dynamics_SRB(t, Xt, Ut, Xd, U_ext, p):
    """Compute dX/dt for the single rigid body model.

    Matches MATLAB dynamics_SRB.m.
    """
    mass = p['mass']
    J = p['J']
    g = 9.81

    Xt = np.asarray(Xt, dtype=float).reshape(-1)
    Ut = np.asarray(Ut, dtype=float).reshape(-1)

    pc = Xt[0:3].reshape(3, 1)
    dpc = Xt[3:6].reshape(3, 1)
    R = Xt[6:15].reshape(3, 3, order='F')
    wb = Xt[15:18].reshape(3, 1)
    pf34 = Xt[18:30].reshape(3, 4, order='F')

    pfd34 = Xd[18:30].reshape(3, 4, order='F')

    r34 = pf34 - np.tile(pc, (1, 4))
    f34 = Ut.reshape(3, 4, order='F')

    # Linear acceleration
    ddpc = (1.0 / mass) * (f34.sum(axis=1).reshape(3, 1) + U_ext.reshape(3, 1)) + np.array([[0.0], [0.0], [-g]])

    # Rotation matrix derivative
    dR = R @ hatMap(wb)

    # Body torque in spatial frame
    tau_s = np.zeros((3, 1))
    for ii in range(4):
        tau_s = tau_s + hatMap(r34[:, ii]) @ f34[:, ii].reshape(3, 1)

    tau_ext = hatMap(R @ p['p_ext']) @ U_ext.reshape(3, 1)
    tau_tot = tau_s + tau_ext
    dwb = np.linalg.solve(J, R.T @ tau_tot - hatMap(wb) @ J @ wb)

    # Foot position PD (swing phase)
    dpf = p['Kp_sw'] * (pfd34.reshape(-1, 1, order='F') - pf34.reshape(-1, 1, order='F'))

    dXdt = np.concatenate([
        dpc.reshape(-1),
        ddpc.reshape(-1),
        dR.flatten(order='F'),
        dwb.reshape(-1),
        dpf.reshape(-1),
    ])
    return dXdt
