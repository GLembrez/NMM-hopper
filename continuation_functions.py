import casadi as cs
import numpy as np
from scipy.integrate import solve_ivp
import hyperparameters as params
import dynamics

stance = lambda t, x: np.array(dynamics.fs(x)).squeeze()
flight = lambda t, x: np.array(dynamics.ff(x)).squeeze()
touch_down = lambda t, x: x[1] - np.cos(x[2])
lift_off = lambda t, x: x[0] ** 2 + x[1] ** 2 - 1
apex = lambda t, x: x[4]
touch_down.terminal = True
touch_down.direction = -1
lift_off.terminal = True
lift_off.direction = 1
apex.terminal = True
apex.direction = -1


def integration(x0, fun, event):
    return solve_ivp(
        fun=fun,
        t_span=(0.0, 10.0),
        y0=x0,
        method="DOP853",
        max_step=5e-2,
        rtol=1e-12,
        atol=1e-12,
        events=(event),
    )


def interpolate(traj, N):
    n = traj.y.shape[0]
    k = 0
    T = np.linspace(0.0, traj.t_events[0], N)
    xh = np.zeros((n, N))
    for i in range(N):
        while T[i] > traj.t[k + 1]:
            k += 1
        a = (T[i] - traj.t[k]) / (traj.t[k + 1] - traj.t[k])
        xh[:, i] = (1 - a) * traj.y[:, k] + a * traj.y[:, k + 1]
    return xh


def init_trajectory(x_apex):
    apex_to_TD = integration(x_apex, flight, touch_down)
    x_TD = apex_to_TD.y_events[0][0]
    x0_TD = np.array([-np.sin(x_TD[2]), np.cos(x_TD[2]), x_TD[3], x_TD[4]])
    TD_to_LO = integration(x0_TD, stance, lift_off)
    x_LO = TD_to_LO.y_events[0][0]
    x0_LO = np.array(dynamics.stance_to_flight(x_LO)).reshape((6,))
    LO_to_apex = integration(x0_LO, flight, apex)

    x1h = interpolate(apex_to_TD, params.N_F)
    x2h = interpolate(TD_to_LO, params.N_S)
    x3h = interpolate(LO_to_apex, params.N_F)

    x1h[0, :] += x2h[0, 0] - x1h[0, -1]
    traj_output = np.hstack([x1h, dynamics.stance_to_flight(x2h), x3h])
    dtf = apex_to_TD.t_events[0] / params.N_F
    dts = TD_to_LO.t_events[0] / params.N_S

    return traj_output, np.concatenate([dtf, dts])


def estimation(traj_current, dt_current, traj_previous, dt_previous):
    dtraj = traj_current - traj_previous
    traj_next = traj_current + dtraj
    dt_next = dt_current + (dt_current - dt_previous)
    return traj_next, dt_next


def register(traj, dt, E, branch):
    branch["traj"].append(traj.copy())
    branch["dt"].append(dt.copy())
    branch["E"].append(E)
