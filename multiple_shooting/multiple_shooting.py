import casadi as cs
import numpy as np
from scipy.linalg import eig
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

np.set_printoptions(precision=3)


### _______________________ CONSTANTS ______________________________________


N_F = 25
N_S = 50
K = 40
W = cs.sqrt(10)


### _______________________ DYNAMICS ______________________________________


xs = cs.SX.sym("xs", 4)
xf = cs.SX.sym("xf", 6)
dt = cs.SX.sym("dt")

l = cs.sqrt(xs[0] ** 2 + xs[1] ** 2)
Es = 0.5 * (xs[2] ** 2 + xs[3] ** 2) + xs[1] + 0.5 * (l - 1) ** 2
Ef = 0.5 * (xf[3] ** 2 + xf[4] ** 2) + xf[1]
dxs = cs.vertcat(xs[2:], K * (1 - l) * xs[0] / l, K * (1 - l) * xs[1] / l - 1)
dxf = cs.vertcat(xf[3:], 0, -1, -(W**2) * xf[2])

ff = cs.Function("ff", [xf], [dxf])
fs = cs.Function("fs", [xs], [dxs])
energy_flight = cs.Function("energy_flight", [xf], [Ef])
energy_stance = cs.Function("energy_stance", [xs], [Es])

k1s = fs(xs)
k1f = ff(xf)
k2s = fs(xs + dt / 2 * k1s)
k2f = ff(xf + dt / 2 * k1f)
k3s = fs(xs + dt / 2 * k2s)
k3f = ff(xf + dt / 2 * k2f)
k4s = fs(xs + dt * k3s)
k4f = ff(xf + dt * k3f)
xs_ = xs + dt / 6 * (k1s + 2 * k2s + 2 * k3s + k4s)
xf_ = xf + dt / 6 * (k1f + 2 * k2f + 2 * k3f + k4f)

RK4s = cs.Function("RK4s", [xs, dt], [xs_])
RK4f = cs.Function("RK4f", [xf, dt], [xf_])
traj_f = RK4f.fold(N_F)
traj_s = RK4s.fold(N_S)

s2f = cs.vertcat(
    xs[:2], cs.atan(-xs[0] / xs[1]), xs[2:], (xs[0] * xs[3] - xs[1] * xs[2]) / l
)
f2s = cs.vertcat(-cs.sin(xf[2]), cs.cos(xf[2]), xf[3:5])
stance_to_flight = cs.Function("stance_to_flight", [xs], [s2f])
flight_to_stance = cs.Function("flight_to_stance", [xf], [f2s])


### _______________________ CONTINUATION SOLVER ______________________________________


opti = cs.Opti()
opti.solver("ipopt", {"print_time": 0, "ipopt.print_level": 0, "ipopt.tol": 1e-12})

energy = opti.parameter()

dts = opti.variable()
dtf = opti.variable()
traj = opti.variable(6, N_F + N_S + N_F)

for i in range(N_F - 1):
    # flight dynamics constraint
    opti.subject_to(traj[:, i + 1] == RK4f(traj[:, i], dtf))
    opti.subject_to(traj[:, N_F + N_S + i + 1] == RK4f(traj[:, N_F + N_S + i], dtf))

for i in range(N_S - 1):
    # stance dynamics constraint
    opti.subject_to(
        traj[:, N_F + i + 1] == stance_to_flight(RK4s(traj[[0, 1, 3, 4], N_F + i], dts))
    )

# touch-down
opti.subject_to(flight_to_stance(traj[:, N_F - 1]) == flight_to_stance(traj[:, N_F]))

# lift-off
opti.subject_to(
    flight_to_stance(traj[:, N_F + N_S - 1]) == flight_to_stance(traj[:, N_F + N_S])
)

# periodicity
opti.subject_to(traj[1:, 0] == traj[1:, -1])
opti.subject_to(traj[0, 0] == 0)
opti.subject_to(traj[0, N_F] == traj[0, N_F - 1])
opti.subject_to(traj[0, N_F + N_S] == traj[0, N_F + N_S - 1])


opti.subject_to(energy_flight(traj[:, 0]) == energy)

### _________________________ OOP FORMULATION ___________________________________


class ContinuationSolver:

    def __init__(self):

        self.opti = cs.opti()
        self.opti.solver(
            "ipopt", {"print_time": 0, "ipopt.print_level": 0, "ipopt.tol": 1e-12}
        )

        self.dt = self.opti.variable(2)
        self.traj = self.opti.variable(6, 2 * N_F + N_S)

        self.declare_constraints()

    def declare_constraints(self):
        for i in range(N_F - 1):
            # flight dynamics constraint
            self.opti.subject_to(
                self.traj[:, i + 1] == RK4f(self.traj[:, i], self.dt[0])
            )
            self.opti.subject_to(
                self.traj[:, N_F + N_S + i + 1]
                == RK4f(self.traj[:, N_F + N_S + i], self.dt[0])
            )

        for i in range(N_S - 1):
            # stance dynamics constraint
            self.opti.subject_to(
                self.traj[:, N_F + i + 1]
                == stance_to_flight(RK4s(self.traj[[0, 1, 3, 4], N_F + i], self.dt[1]))
            )

        # touch-down
        self.opti.subject_to(
            flight_to_stance(self.traj[:, N_F - 1])
            == flight_to_stance(self.traj[:, N_F])
        )

        # lift-off
        self.opti.subject_to(
            flight_to_stance(self.traj[:, N_F + N_S - 1])
            == flight_to_stance(self.traj[:, N_F + N_S])
        )

        # periodicity
        self.opti.subject_to(self.traj[1:, 0] == self.traj[1:, -1])
        self.opti.subject_to(self.traj[0, 0] == 0)
        self.opti.subject_to(self.traj[0, N_F] == self.traj[0, N_F - 1])
        self.opti.subject_to(self.traj[0, N_F + N_S] == self.traj[0, N_F + N_S - 1])


### _______________________ INITIALISATION ______________________________________

stance = lambda t, x: np.array(fs(x)).squeeze()
flight = lambda t, x: np.array(ff(x)).squeeze()
touch_down = lambda t, x: x[1] - np.cos(x[2])
lift_off = lambda t, x: x[0] ** 2 + x[1] ** 2 - 1
apex = lambda t, x: x[4]
touch_down.terminal = True
touch_down.direction = -1
lift_off.terminal = True
lift_off.direction = 1
apex.terminal = True
apex.direction = -1


def solver(x0, fun, event):
    return solve_ivp(
        fun=fun,
        t_span=(0.0, 10.0),
        y0=x0,
        method="DOP853",
        rtol=1e-12,
        atol=1e-12,
        events=(event),
        vectorized=True,
    )


def init_trajectory(x0):
    sol1 = solver(x0, flight, touch_down)
    y1 = sol1.y_events[0][0]
    x02 = np.array([-np.sin(y1[2]), np.cos(y1[2]), y1[3], y1[4]])
    sol2 = solver(x02, stance, lift_off)
    y2 = sol2.y_events[0][0]
    x03 = np.array(
        [
            y2[0],
            y2[1],
            np.arctan(-y2[0] / y2[1]),
            y2[2],
            y2[3],
            y2[0] * y2[3] - y2[1] * y2[2],
        ]
    )
    sol3 = solver(x03, flight, apex)
    T1 = np.linspace(0, sol1.t_events[0], N_F)
    T2 = np.linspace(0, sol2.t_events[0], N_S)
    T3 = np.linspace(0, sol3.t_events[0], N_F)
    x1h, x2h, x3h = np.zeros((6, N_F)), np.zeros((4, N_S)), np.zeros((6, N_F))
    i1, i2, i3 = 0, 0, 0
    for i in range(N_F):
        if T1[i] > sol1.t[i1 + 1]:
            i1 += 1
        if T3[i] > sol3.t[i3 + 1]:
            i3 += 1
        alpha1 = (T1[i] - sol1.t[i1]) / (sol1.t[i1 + 1] - sol1.t[i1])
        alpha3 = (T3[i] - sol3.t[i3]) / (sol3.t[i3 + 1] - sol3.t[i3])
        x1h[:, i] = (1 - alpha1) * sol1.y[:, i1] + alpha1 * sol1.y[:, i1 + 1]
        x3h[:, i] = (1 - alpha3) * sol3.y[:, i3] + alpha3 * sol3.y[:, i3 + 1]
    for i in range(N_S):
        if T2[i] > sol2.t[i2 + 1]:
            i2 += 1
        alpha2 = (T2[i] - sol2.t[i2]) / (sol2.t[i2 + 1] - sol2.t[i2])
        x2h[:, i] = (1 - alpha2) * sol2.y[:, i2] + alpha2 * sol2.y[:, i2 + 1]
    traj = cs.horzcat(x1h, stance_to_flight(x2h), x3h)
    dtf = sol1.t_events[0] / N_F
    dts = sol2.t_events[0] / N_S

    return traj, dtf, dts


x0 = np.array([0.0, 1.1, 0.0, 0.0, 0.0, 0.0])
traj_eval, dtf_eval, dts_eval = init_trajectory(x0)

opti.set_initial(traj, traj_eval)
opti.set_initial(dts, dts_eval)
opti.set_initial(dtf, dtf_eval)
opti.set_value(energy, energy_flight(traj_eval[:, 0]))

opti.solve()

traj_output = opti.value(traj)
plt.figure()
plt.plot(traj_output[1, :].T, traj_output[4, :].T)
plt.show()


## TODO
# 1. Implement distance to previous + increasing energy constraint | remove energy constraint
# 2. transform into class ContinuationSolver()
# 3. implement continuation functions (monodromy, floquet, etc)
# 4. add distance to explored branch constraint
# 5. profit.
