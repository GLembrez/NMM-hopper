import casadi as cs
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
import hyperparameters as params
import simple_shooting as ss
import dynamics

stance = lambda t, x, K: np.array(dynamics.fs(x, 0, K)).squeeze()
flight = lambda t, x, W: np.array(dynamics.ff(x, 0, W)).squeeze()
touch_down = lambda t, x, arg: x[1] - np.cos(x[2])
lift_off = lambda t, x, arg: x[0] ** 2 + x[1] ** 2 - 1
apex = lambda t, x, arg: x[4]
touch_down.terminal = True
touch_down.direction = -1
lift_off.terminal = True
lift_off.direction = 1
apex.terminal = False
apex.direction = -1


def solver(x0, fun, events, args):
    return solve_ivp(
        fun=fun,
        t_span=(0.0, 10.0),
        y0=x0,
        max_step=1e-1,
        method="RK45",
        rtol=1e-12,
        atol=1e-12,
        events=events,
        args=args,
        vectorized=True,
    )


def init_trajectory(x0, K, W):
    sol1 = solver(x0, flight, (touch_down, apex), (W,))
    y1 = sol1.y_events[0][0]
    x02 = np.array([-np.sin(y1[2]), np.cos(y1[2]), y1[3], y1[4]])
    sol2 = solver(x02, stance, (lift_off,), (K,))
    T1 = np.linspace(0, sol1.t_events[0][0], params.N_F)
    T2 = np.linspace(0, sol2.t_events[0], params.N_S)
    x1h, x2h = np.zeros((6, params.N_F)), np.zeros((4, params.N_S))
    i1, i2 = 0, 0
    for i in range(params.N_F):
        while T1[i] > sol1.t[i1 + 1]:
            i1 += 1
        alpha1 = (T1[i] - sol1.t[i1]) / (sol1.t[i1 + 1] - sol1.t[i1])
        x1h[:, i] = (1 - alpha1) * sol1.y[:, i1] + alpha1 * sol1.y[:, i1 + 1]
    for i in range(params.N_S):
        while T2[i] > sol2.t[i2 + 1]:
            i2 += 1
        alpha2 = (T2[i] - sol2.t[i2]) / (sol2.t[i2 + 1] - sol2.t[i2])
        x2h[:, i] = (1 - alpha2) * sol2.y[:, i2] + alpha2 * sol2.y[:, i2 + 1]
    return x1h, x2h, sol1.t_events[0][0] / params.N_F, sol2.t_events[0] / params.N_S


def energy_flight(x):
    return 0.5 * (x[3] ** 2 + x[4] ** 2) + x[1]


def energy_apex(x):
    return 0.5 * (x[2] ** 2) + x[0]


def initialize_next(traj_compact, apex, dir_old, dist, K, W):
    dir1, dir2, multiplier = ss.compute_multipliers(traj_compact, apex, dir_old, K, W)
    if energy_apex(apex[[1, 2, 3, 5]] + dist * dir1) < energy_apex(apex[[1, 2, 3, 5]]):
        dist = -dist
    x0 = np.array(apex).reshape((6,1))
    x0[[1, 2, 3, 5],:] += dist * dir1.reshape((4,1))
    var = ss.full_newton(traj_compact, x0, K, W)
    xfh, xsh, dtf, dts = ss.trajectory(var, K, W)
    return xfh, xsh, dtf, dts, multiplier, dir1, dir2
