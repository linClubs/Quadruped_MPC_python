"""Parameters container.

Python translation of get_params.m.

We use a plain dict to mirror the MATLAB struct ``p``. Behaviour is identical
to the MATLAB version, including the per-gait overrides.
"""

import numpy as np


def get_params(gait):
    """Return a dict of parameters for the requested gait.

    gait codes:
        0 - trot, 1 - bound, 2 - pacing, 3 - gallop,
        4 - trot run, 5 - crawl
    """
    p = {}
    p['predHorizon'] = 6
    p['simTimeStep'] = 1.0 / 200.0
    p['Tmpc'] = 4.0 / 100.0
    p['gait'] = gait
    p['Umax'] = 50.0
    p['decayRate'] = 1.0
    p['freq'] = 30
    p['Rground'] = np.eye(3)
    p['Qf'] = np.diag([1e5, 2e5, 3e5, 5e2, 1e3, 150, 1e3, 1e4, 800, 40, 40, 10])

    if gait == 1:                # bound
        p['Tst'] = 0.1
        p['Tsw'] = 0.18
        p['predHorizon'] = 7
        p['simTimeStep'] = 1.0 / 100.0
        p['Tmpc'] = 2.0 / 100.0
        p['decayRate'] = 1.0
        p['R'] = np.diag(np.tile([0.1, 0.1, 0.1], (4, 1)).reshape(-1))
        p['Q'] = np.diag([5e4, 2e4, 1e6, 4e3, 5e2, 5e2, 1e4, 5e4, 1e3, 1e2, 5e2, 1e2])
        p['Qf'] = np.diag([2e5, 5e4, 5e6, 8e3, 5e2, 5e2, 1e4, 5e4, 5e3, 1e2, 1e2, 1e2])
    elif gait == 2:              # pacing
        p['Tst'] = 0.12
        p['Tsw'] = 0.12
        p['R'] = np.diag(np.tile([0.1, 0.2, 0.1], (4, 1)).reshape(-1))
        p['Q'] = np.diag([5e3, 5e3, 9e4, 5e2, 5e2, 5e2, 7e3, 7e3, 7e3, 5e1, 5e1, 5e1])
    elif gait == 3:              # gallop
        p['Tst'] = 0.08
        p['Tsw'] = 0.2
        p['R'] = np.diag(np.tile([0.1, 0.2, 0.1], (4, 1)).reshape(-1))
        p['Q'] = np.diag([3e3, 3e3, 4e6, 5e2, 1e3, 150, 1e4, 1e4, 800, 1e2, 5e1, 5e1])
    elif gait == 4:              # trot run
        p['Tst'] = 0.12
        p['Tsw'] = 0.2
        p['Tmpc'] = 3.0 / 100.0
        p['predHorizon'] = 6
        p['decayRate'] = 1.0
        p['R'] = np.diag(np.tile([0.1, 0.18, 0.08], (4, 1)).reshape(-1))
        p['Q'] = np.diag([1e5, 1e5, 1e5, 1e3, 1e3, 1e3, 2e3, 1e4, 800, 100, 40, 10])
        p['Qf'] = np.diag([1e5, 1.5e5, 2e4, 1.5e3, 1e3, 100, 2e3, 2e3, 800, 100, 60, 10])
    elif gait == 5:              # crawl
        p['Tst'] = 0.3
        p['Tsw'] = 0.1
        p['R'] = np.diag(np.tile([0.1, 0.2, 0.1], (4, 1)).reshape(-1))
        p['Q'] = np.diag([5e5, 5e5, 9e5, 5, 5, 5, 3e3, 3e3, 3e3, 3, 3, 3])
    else:                         # trot (default)
        p['predHorizon'] = 6
        p['simTimeStep'] = 1.0 / 100.0
        p['Tmpc'] = 8.0 / 100.0
        p['Tst'] = 0.3
        p['Tsw'] = 0.15
        p['R'] = np.diag(np.tile([0.1, 0.2, 0.1], (4, 1)).reshape(-1))
        p['Q'] = np.diag([1e5, 2e5, 3e5, 5e2, 1e3, 1e3, 1e3, 1e4, 800, 40, 40, 10])
        p['Qf'] = p['Q']

    # --- Physical Parameters ---
    p['mass'] = 5.5
    p['J'] = np.diag([0.026, 0.112, 0.075])
    p['g'] = 9.81
    p['mu'] = 1.0
    p['z0'] = 0.2
    p['pf34'] = np.array([
        [0.15, 0.15, -0.15, -0.15],
        [0.094, -0.094, 0.094, -0.094],
        [0.0, 0.0, 0.0, 0.0],
    ])

    p['L'] = 0.301
    p['W'] = 0.088
    p['d'] = 0.05
    p['h'] = 0.05
    p['l1'] = 0.14
    p['l2'] = 0.14

    p['Kp_sw'] = 300.0

    p['body_color'] = np.array([42, 80, 183]) / 255.0
    p['leg_color'] = np.array([7, 179, 128]) / 255.0
    p['ground_color'] = np.array([195, 232, 243]) / 255.0

    return p
