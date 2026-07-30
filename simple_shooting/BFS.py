import casadi as cs
import numpy as np
from collections import deque
from matplotlib import pyplot as plt
from scipy.linalg import null_space
from time import time
from copy import deepcopy

np.set_printoptions(precision=3, suppress=True)

N_F = 25
N_S = 50
K = 40
W = cs.sqrt(10)
N_BRANCH = 5
N_STEPS = 1000
N_NEWTON = 2
N_BIFUR = 10
STEP_SIZE = 0.01


def process_bifurcation(_u, _J, p):
    c1 = -deepcopy(p)
    val, vec = np.linalg.eig(cs.vertcat(_J, c1.T))
    idx_val = np.argmin(np.abs(val))
    c2 = vec[:, idx_val].real.reshape((11,1))
    c2 =  c2 - np.dot(c1.T,c2) * c1 
    c2 = c2 / np.linalg.norm(c2)
    val, vec = np.linalg.eig(_J @ _J.T)
    idx_val = np.argmin(np.abs(val))
    e = vec[:, idx_val].real
    b12 = hessian(_u, c1, c2, e)
    b22 = hessian(_u, c2, c2, e)
    beta2 = 1
    beta1 = -b22 / (2 * b12)
    c_star = beta1 * c1 + beta2 * c2
    c_star = c_star / np.linalg.norm(c_star)
    return deque([(_u, -c_star),(_u, c1)])


def explore_branch(BP, to_explore):
    branch = []
    p_list = []
    bifurcation_step = 0
    _u, p = BP
    p_next = null_space(compute_J(_u + 0.001 * p))
    d = STEP_SIZE if p[-1] > 0 else - STEP_SIZE
    print(np.array(p).reshape((1,11)))

    for step in range(N_STEPS):

        if step>5 and p_list[-2].T @ p_list[-1] < 0:
            d = 0.5 * d
            bifurcation_step += 1

        # d = - np.abs(d) if p[-1]<0 else np.abs(d)
        p_list.append(p)
        branch.append(_u)
        _u += d * p 
        _z,_E = _u[:10], _u[10]
        z_star = newton(_z,_E)
        _u = cs.vertcat(z_star,_E)
        _J = compute_J(_u)
        p = null_space(_J)

        if np.linalg.det(cs.vertcat(_J,p.T)) < 0:
            p = -p

        if bifurcation_step == N_BIFUR:
            initial_conditions = process_bifurcation(_u,_J,p)
            for ic in initial_conditions:
                to_explore.append(ic)
            return branch, to_explore

    return branch, to_explore


xs = cs.SX.sym("xs", 4)
xf = cs.SX.sym("xf", 6)
xi = cs.SX.sym("xi")
dt = cs.SX.sym("dt")
x_init = cs.SX.sym("x_init", 6)
t = cs.SX.sym("t", 3)
E = cs.SX.sym("E")
du1 = cs.SX.sym("du1", 11)
du2 = cs.SX.sym("du2", 11)
phi = cs.SX.sym("phi", 10)

z = cs.vertcat(x_init, t, xi)
u = cs.vertcat(z, E)

l = cs.sqrt(xs[0] ** 2 + xs[1] ** 2)
Es = 0.5 * (xs[2] ** 2 + xs[3] ** 2) + xs[1] + 0.5 * (l - 1) ** 2
Ef = 0.5 * (xf[3] ** 2 + xf[4] ** 2) + xf[1]
dxs = cs.vertcat(
    xs[2:], K * (1 - l) * xs[0] / l, K * (1 - l) * xs[1] / l - 1
) + xi * cs.gradient(Es, xs)
dxf = cs.vertcat(xf[3:], 0, -1, -(W**2) * xf[2]) + xi * cs.gradient(Ef, xf)

ff = cs.Function("ff", [xf, xi], [dxf])
fs = cs.Function("fs", [xs, xi], [dxs])
energy_flight = cs.Function("energy_flight", [xf], [Ef])
energy_stance = cs.Function("energy_stance", [xs], [Es])

k1s = fs(xs, xi)
k1f = ff(xf, xi)
k2s = fs(xs + dt / 2 * k1s, xi)
k2f = ff(xf + dt / 2 * k1f, xi)
k3s = fs(xs + dt / 2 * k2s, xi)
k3f = ff(xf + dt / 2 * k2f, xi)
k4s = fs(xs + dt * k3s, xi)
k4f = ff(xf + dt * k3f, xi)
xs_ = xs + dt / 6 * (k1s + 2 * k2s + 2 * k3s + k4s)
xf_ = xf + dt / 6 * (k1f + 2 * k2f + 2 * k3f + k4f)

s2f = cs.vertcat(
    xs[:2], cs.atan(-xs[0] / xs[1]), xs[2:], (xs[0] * xs[3] - xs[1] * xs[2]) / l
)
f2s = cs.vertcat(-cs.sin(xf[2]), cs.cos(xf[2]), xf[3:5])

stance_to_flight = cs.Function("stance_to_flight", [xs], [s2f])
flight_to_stance = cs.Function("flight_to_stance", [xf], [f2s])
RK4s = cs.Function("RK4s", [xs, dt, xi], [xs_])
RK4f = cs.Function("RK4f", [xf, dt, xi], [xf_])

traj_f = RK4f.fold(N_F)
traj_s = RK4s.fold(N_S)

x_TD = traj_f(x_init, t[0], xi)
x_LO = traj_s(flight_to_stance(x_TD), t[1], xi)
x_apex = traj_f(stance_to_flight(x_LO), t[2], xi)

root = cs.vertcat(
    x_apex[1:] - x_init[1:],
    x_init[0],
    x_init[4],
    energy_flight(x_init) - E,
    cs.cos(x_TD[2]) - x_TD[1],
    x_LO[0] ** 2 + x_LO[1] ** 2 - 1,
)

root_rev = cs.vertcat(
    x_apex[[1, 4]] - x_init[[1, 4]],
    x_apex[[2, 3, 5]] + x_init[[2, 3, 5]],
    x_init[0],
    x_init[4],
    energy_flight(x_init) - E,
    cs.cos(x_TD[2]) - x_TD[1],
    x_LO[0] ** 2 + x_LO[1] ** 2 - 1,
)

J_root = cs.jacobian(root, z)
J_ext = cs.jacobian(root, u)
J_root_rev = cs.jacobian(root_rev, z)
J_ext_rev = cs.jacobian(root_rev, u)
z_newton = z - cs.inv(J_root) @ root
z_newton_rev = z - cs.inv(J_root_rev) @ root_rev

b = du1.T @ cs.jtimes(J_ext, u, du2).T @ phi

newton_step = cs.Function("newton_step", [z, E], [z_newton])
newton_step_rev = cs.Function("newton_step_rev", [z, E], [z_newton_rev])
compute_J = cs.Function("compute_J", [u], [J_ext])
compute_J_rev = cs.Function("compute_J_rev", [u], [J_ext_rev])
residual = cs.Function("residual", [u], [root])
residual_rev = cs.Function("residual_rev", [u], [root_rev])
hessian = cs.Function("Hessian", [u, du1, du2, phi], [b])

newton = newton_step.fold(N_NEWTON)
newton_rev = newton_step_rev.fold(N_NEWTON)


to_explore = deque()
idx_branch = 0
connected_component = []

_z = np.array([0.0, 1.01, 0.0, 0.0, 0.0, 0.0, 0.1, 0.01, 0.1, 0.0])
_E = _z[1]
_z_star = newton(_z, _E)
_u = cs.vertcat(_z_star, _E)
_J = compute_J(_u)
p = null_space(_J)
to_explore.appendleft((_u, p))

while to_explore and idx_branch < N_BRANCH:
    idx_branch += 1
    bifurcation_point = to_explore.popleft()
    branch, to_explore = explore_branch(bifurcation_point, to_explore)
    connected_component.append(branch)

fig = plt.figure()
ax = fig.add_subplot(1,1,1,projection="3d")
for i in range(len(connected_component)):
    branch = np.array(connected_component[i])
    tangent = np.array(branch[2]-branch[1])
    print((tangent/np.linalg.norm(tangent)).reshape((1,11)))
    plt.plot(branch[:,1],branch[:,2],branch[:,3])
plt.show()

print(connected_component[0][0],connected_component[0][1],connected_component[0][2])


# TODO verify orthogonality of span ker(J)
