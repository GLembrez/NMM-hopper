import casadi as cs
import numpy as np
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
epsilon = cs.SX.sym("epsilon")
dt = cs.SX.sym("dt")

l = cs.sqrt(xs[0] ** 2 + xs[1] ** 2)
Es = 0.5 * (xs[2] ** 2 + xs[3] ** 2) + xs[1] + 0.5 * (l - 1) ** 2
Ef = 0.5 * (xf[3] ** 2 + xf[4] ** 2) + xf[1]
dxs = cs.vertcat(xs[2:], K * (1 - l) * xs[0] / l, K * (1 - l) * xs[1] / l - 1) + epsilon * cs.gradient(Es,xs)
dxf = cs.vertcat(xf[3:], 0, -1, -(W**2) * xf[2]) + epsilon * cs.gradient(Ef,xf)

ff = cs.Function("ff", [xf,epsilon], [dxf])
fs = cs.Function("fs", [xs,epsilon], [dxs])
energy_flight = cs.Function("energy_flight", [xf], [Ef])
energy_stance = cs.Function("energy_stance", [xs], [Es])

k1s = fs(xs,epsilon)
k1f = ff(xf,epsilon)
k2s = fs(xs + dt / 2 * k1s,epsilon)
k2f = ff(xf + dt / 2 * k1f,epsilon)
k3s = fs(xs + dt / 2 * k2s,epsilon)
k3f = ff(xf + dt / 2 * k2f,epsilon)
k4s = fs(xs + dt * k3s,epsilon)
k4f = ff(xf + dt * k3f,epsilon)
xs_ = xs + dt / 6 * (k1s + 2 * k2s + 2 * k3s + k4s)
xf_ = xf + dt / 6 * (k1f + 2 * k2f + 2 * k3f + k4f)

RK4s = cs.Function("RK4s", [xs, dt,epsilon], [xs_])
RK4f = cs.Function("RK4f", [xf, dt,epsilon], [xf_])
traj_f = RK4f.fold(N_F)
traj_s = RK4s.fold(N_S)

s2f = cs.vertcat(
    xs[:2], cs.atan(-xs[0] / xs[1]), xs[2:], (xs[0] * xs[3] - xs[1] * xs[2]) / l
)
f2s = cs.vertcat(-cs.sin(xf[2]), cs.cos(xf[2]), xf[3:5])
stance_to_flight = cs.Function("stance_to_flight", [xs], [s2f])
flight_to_stance = cs.Function("flight_to_stance", [xf], [f2s])


### _________________________ OOP FORMULATION ___________________________________



class ContinuationSolver():

    def __init__(self,periodic):

        self.periodic = periodic
        self.opti = cs.Opti()
        self.opti.solver(
            "ipopt", {"print_time": 0, "ipopt.print_level": 0, "ipopt.tol": 1e-12}
        )

        self.dt = self.opti.variable(2)
        self.traj = self.opti.variable(6, 2 * N_F + N_S)

        self.epsilon = self.opti.variable()
        self.Ed = self.opti.parameter()

        self.declare_constraints()
        self.set_periodicity()

    def declare_constraints(self):
        for i in range(N_F - 1):
            # flight dynamics constraint
            self.opti.subject_to(
                self.traj[:, i + 1] == RK4f(self.traj[:, i], self.dt[0], self.epsilon)
            )
            self.opti.subject_to(
                self.traj[:, N_F + N_S + i + 1]
                == RK4f(self.traj[:, N_F + N_S + i], self.dt[0], self.epsilon)
            )

        for i in range(N_S - 1):
            # stance dynamics constraint
            self.opti.subject_to(
                self.traj[:, N_F + i + 1]
                == stance_to_flight(RK4s(self.traj[[0, 1, 3, 4], N_F + i], self.dt[1], self.epsilon))
            )

        # touch-down
        self.opti.subject_to(self.traj[:, N_F - 1] == self.traj[:,N_F]) # continuity of angular velocity ?
        self.opti.subject_to(self.traj[1,N_F-1] == cs.cos(self.traj[2,N_F-1]))

        # lift-off
        self.opti.subject_to(self.traj[:, N_F + N_S - 1] == self.traj[:, N_F + N_S])
        self.opti.subject_to(self.traj[0,N_F+N_S-1]**2 + self.traj[1,N_F+N_S-1]**2  == 1)

        # apex 
        self.opti.subject_to(self.traj[4,0]==0)

        self.opti.subject_to(energy_flight(self.traj[:, 0]) == self.Ed)

        self.opti.minimize(self.epsilon.T @ self.epsilon)

    def set_periodicity(self):
        if self.periodic:
            # periodicity
            self.opti.subject_to(self.traj[[1,2,3,5], 0] == self.traj[[1,2,3,5], -1])
            # self.opti.subject_to(self.traj[0,0] == 0)
        else:
            self.opti.subject_to(self.traj[1,0] == self.traj[1,-1])
            self.opti.subject_to(self.traj[[2,3,5],0] == -self.traj[[2,3,5],-1])
            # self.opti.subject_to(self.traj[2,0]==0)

    def initialize(self, traj, dt, Ed):
        self.opti.set_initial(self.traj, traj)
        self.opti.set_initial(self.dt, dt)
        self.opti.set_value(self.Ed, Ed)
        self.opti.set_initial(self.epsilon, 0.0)

    def solve(self):
        self.opti.solve()
        return (
            self.opti.value(self.traj),
            self.opti.value(self.dt),
            self.opti.value(self.Ed),
        )


### ________________________ BIFURCATIONS ______________________________________

reverse_matrix = cs.DM([[1,0,0,0],[0,0,0,0],[0,0,-1,0],[0,0,0,-1]])

init = cs.SX.sym("init", 5)
apex = cs.SX.sym("apex", 5)
times = cs.SX.sym("times", 2)
z = cs.vertcat(apex, times)

x_TD = traj_f(cs.vertcat(0.0,init), times[0],0.0)
x_LO = traj_s(flight_to_stance(x_TD), times[1],0.0)
x_a = traj_f(stance_to_flight(x_LO), times[0],0.0)

R = cs.vertcat(
    x_a[1:] - apex,
    cs.cos(x_TD[2]) - x_TD[1],
    x_LO[0] ** 2 + x_LO[1] ** 2 - 1,
)
jac_R = cs.jacobian(R, z)
newton = z - cs.inv(jac_R) @ R
jac_newton = cs.jacobian(newton[0, 1, 2, 4] - init[0, 1, 2, 4], init[0, 1, 2, 4])
jac_newton_reverse = cs.jacobian(newton[0, 1, 2, 4] - reverse_matrix @ init[0, 1, 2, 4],init[0, 1, 2, 4])

eval_newton = cs.Function("eval_newton", [z, init], [newton])
full_newton = eval_newton.fold(10)
monodromy = cs.Function(
    "monodromy", [init, apex, times], [jac_newton]
)
monodromy_reverse = cs.Function(
    "monodromy_reverse", [init, apex, times], [jac_newton_reverse]
)



def estimation(traj_current, dt_current, traj_previous, dt_previous, d=1):
    dtraj = traj_current - traj_previous
    traj_next = traj_current + dtraj
    dt_next = dt_current + (dt_current - dt_previous)
    return traj_next, dt_next, energy_flight(traj_next[:, 0])


def register(traj, dt, E, branch):
    branch["traj"].append(traj.copy())
    branch["dt"].append(dt.copy())
    branch["E"].append(E)


### _______________________ INITIALISATION ______________________________________


stance = lambda t, x: np.array(fs(x,0.0)).squeeze()
flight = lambda t, x: np.array(ff(x,0.0)).squeeze()
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
        max_step=1e-1,
        rtol=1e-12,
        atol=1e-12,
        events=(event),
    )


def init_trajectory(x0):
    sol1 = integration(x0, flight, touch_down)
    y1 = sol1.y_events[0][0]
    x02 = np.array([-np.sin(y1[2]), np.cos(y1[2]), y1[3], y1[4]])
    sol2 = integration(x02, stance, lift_off)
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
    sol3 = integration(x03, flight, apex)
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
    x1h[0,:] += x2h[0,0] - x1h[0,-1]
    traj_output = np.hstack([x1h, stance_to_flight(x2h), x3h])
    dtf = sol1.t_events[0] / N_F
    dts = sol2.t_events[0] / N_S

    return traj_output, np.concatenate([dtf, dts])


###___________________________ CONTINUATION ______________________________________________
from tqdm import tqdm


fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# # for y0 in [1.12359,1.27227]:
x0 = np.array([0.0, 1.01, 0.0, 0.0, 0.0, 0.0]) # 1.830 1.225
traj_eval, dt_eval = init_trajectory(x0)
N = 400
distance = 0.01
epsilon = 0.15
periodic = False
solver = ContinuationSolver(periodic)


from collections import deque
to_explore = deque()
to_explore.append((traj_eval,dt_eval,np.array([1.0,0.0,0.0,0.0])))
components = []
idx_branch = 0
n_branch = 1
floquet = []
energy = []

while to_explore and idx_branch<n_branch:
    dE = 1e-6
    previous_mul = None
    branch = {"traj": [], "E": [], "dt": []}
    traj_eval,dt_eval,v = to_explore.popleft()
    current_bifurcation = traj_eval[1:,0]
    E = energy_flight(traj_eval[:,0])
    register(traj_eval, dt_eval, E, branch)
    x0 = traj_eval[:,0].copy()
    x0[[1,2,3,5]] += 1e-3 * np.real(v)
    traj_eval, dt_eval = init_trajectory(x0)
    solver.initialize(traj_eval, dt_eval, branch["E"][-1] + dE)
    traj_eval, dt_eval, E = solver.solve()
    register(traj_eval, dt_eval, E, branch)
    print("exploring branch {}".format(idx_branch))
    idx_branch += 1

    for step in tqdm(range(N)):
        traj_eval, dt_eval, E = estimation(
            branch["traj"][-1], branch["dt"][-1], branch["traj"][-2], branch["dt"][-2], distance
        )
        solver.initialize(traj_eval, dt_eval, branch["E"][-1]+dE)
        try:
            traj_eval, dt_eval, E = solver.solve()
            z = full_newton(cs.vertcat(traj_eval[1:,0], dt_eval), traj_eval[1:,0])
            M = monodromy(traj_eval[1:,-1], z[:5], z[5:]) if periodic else monodromy_reverse(traj_eval[1:,-1], z[:5], z[5:])
            values, vectors = np.linalg.eig(M)
            floquet.append(values)
            energy.append(E)
        except:
            print('end of branch reached')
            break
        if (traj_eval[1,:] < 0).any():
            print('end of branch reached')
            break
        register(traj_eval, dt_eval, E, branch)
        idx_sorted = np.argsort(np.abs(values))

        dist_to_bifurcation = np.linalg.norm(current_bifurcation - traj_eval[1:,0])
        dE = np.clip(min(0.1*np.abs(values[idx_sorted[1]]),0.1*dist_to_bifurcation),1e-6,5e-3)

        if np.abs(values[idx_sorted[1]])<epsilon and dist_to_bifurcation > 0.1:
            if previous_mul == None:
                previous_mul = np.abs(values[idx_sorted[1]])
            if previous_mul * (values[idx_sorted[1]].real) < 0 and np.abs(values[idx_sorted[1]].imag)<1e-3 :
                print(previous_mul, values)
                coeff = np.real(previous_mul/(previous_mul - values[idx_sorted[1]]))
                traj_bifurcation = coeff * branch["traj"][-1] + (1-coeff)*branch["traj"][-2]
                dt_bifurcation = coeff * branch["dt"][-1] + (1-coeff) * branch["dt"][-2]
                to_explore.append((traj_bifurcation,dt_bifurcation,vectors[:,idx_sorted[0]]))
                to_explore.append((traj_bifurcation,dt_bifurcation,vectors[:,idx_sorted[1]]))
                print("bifurcation reached at height {}".format(traj_bifurcation[1,0]))
                print(dist_to_bifurcation)
                break
            previous_mul =  np.abs(values[idx_sorted[1]])
    
    components.append(branch)


# plt.plot(energy,[f.real for f in floquet])
plt.plot([f.real for f in floquet], [f.imag for f in floquet])
# for branch in components:
#     print(branch["E"][0],branch["E"][1])
#     ax.plot(np.array([T[1,0] for T in branch["traj"]]),np.array([T[2,0] for T in branch["traj"]]),np.array([T[3,0] for T in branch["traj"]]), marker='.',linewidth=0)
#     T = branch["traj"][-1]
#     ax.plot(np.array(T[1, :]), np.array(T[2, :]),np.array(T[3, :]))
plt.show()
