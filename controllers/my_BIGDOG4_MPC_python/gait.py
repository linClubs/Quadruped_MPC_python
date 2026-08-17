import numpy as np


def set_iterations(n_iterations, current_iteration, iterations_between_mpc):
    """对应 MATLAB setIterations.m"""
    iteration = int(np.floor((current_iteration / iterations_between_mpc) % n_iterations))
    phase = (current_iteration % (iterations_between_mpc * n_iterations)) / (iterations_between_mpc * n_iterations)
    return iteration, phase


def get_swing_state(phase, offsets_float, durations_float):
    """对应 MATLAB getSwingState.m"""
    swing_offset = offsets_float + durations_float
    for i in range(4):
        if swing_offset[i] > 1:
            swing_offset[i] -= 1

    swing_duration = 1.0 - durations_float
    swingstate = phase - swing_offset

    for i in range(4):
        if swingstate[i] < 0:
            swingstate[i] += 1
        if swingstate[i] > swing_duration[i]:
            swingstate[i] = 0
        else:
            swingstate[i] = swingstate[i] / swing_duration[i]
    return swingstate


def get_mpc_table(iteration, n_iterations, offsets, durations):
    """对应 MATLAB getMpcTable.m"""
    mpctable = np.zeros(4 * n_iterations)
    for i in range(n_iterations):
        it = int((i + iteration) % n_iterations)
        progress = it - offsets
        for j in range(4):
            if progress[j] < 0:
                progress[j] += n_iterations
            if progress[j] < durations[j]:
                mpctable[i * 4 + j] = 1
            else:
                mpctable[i * 4 + j] = 0
    return mpctable
