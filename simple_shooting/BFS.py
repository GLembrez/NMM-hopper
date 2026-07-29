import casadi as cs
import numpy as np
from collections import deque
from matplotlib import pyplot as plt
from scipy.linalg import null_space
from time import time 
from copy import deepcopy

np.set_printoptions(precision=9)



N_F = 25
N_S = 50
K = 40
W = cs.sqrt(10)

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


x0 = cs.SX.sym('x0',6)
t = cs.SX.sym('t',3)
xi = cs.SX.sym('xi')
E = cs.SX.sym('E')
z = cs.vertcat(x0,t,xi)
u = cs.vertcat(z,E)

x_TD = traj_f(x0, t[0],xi)
x_LO = traj_s(flight_to_stance(x_TD), t[1],xi)
x_a = traj_f(stance_to_flight(x_LO), t[2],xi)

root = cs.vertcat(
    x_a[1:] - x0[1:],
    x0[0],
    x0[4],
    energy_flight(x0) - E,
    cs.cos(x_TD[2]) - x_TD[1],
    x_LO[0] ** 2 + x_LO[1] ** 2 - 1
)

R_bar = cs.jacobian(root, z)
R_tilde = cs.jacobian(root,u)

z_newton = z - cs.inv(R_bar) @ root 
newton_step = cs.Function("newton_step",[z,E],[z_newton])
compute_R = cs.Function("compute_R",[u], [R_tilde])
J = cs.Function("J",[z],[R_bar])
F = cs.Function("F",[u],[root])

du1 = cs.SX.sym("du1",11)
du2 = cs.SX.sym("du2",11)
phi = cs.SX.sym("phi",10)
b = du1.T @ cs.jtimes(R_tilde, u, du2).T @ phi
hessian = cs.Function('Hessian',[u,du1,du2,phi],[b])

n_branch = 3
z0 = np.array([0.0,1.01,0.,0.,0.,0.,0.1,0.01,0.1,0.0])
E0 = z0[1]
z_star = newton_step(z0,E0)
R_eval = compute_R(cs.vertcat(z_star,z_star[1]))
p = null_space(R_eval)
to_explore = deque()
idx_branch=0
connected_component = []
to_explore.appendleft((cs.vertcat(z_star,E0),p,1))

while to_explore and idx_branch<n_branch:
    branch = []
    p_list = []
    u_next,p,sign = to_explore.popleft()
    idx_branch += 1
    precision = 0
    d = np.sign(sign) * 0.005

    for i in range(1000):
        if np.linalg.det(cs.vertcat(R_eval,p.T)) < 0:
            p = -p

        branch.append(cs.vertcat(z_star,E0))
        p_list.append(p)
        u_next = cs.vertcat(z_star,E0) +  d * p
        z0 = u_next[:10]
        E0 = u_next[10]
        z_star = newton_step(z0,E0)
        R_eval = compute_R(cs.vertcat(z_star,z_star[1]))
        p = null_space(R_eval)


        if precision == 10:
            connected_component.append(branch)
            c1 = -deepcopy(p)
            val,vec = np.linalg.eig(cs.vertcat(R_eval,c1.T))
            idx_val = np.argmin(np.abs(val))
            c2 = vec[:,idx_val].real
            val,vec = np.linalg.eig(R_eval@R_eval.T)
            idx_val = np.argmin(np.abs(val))
            e = vec[:,idx_val].real

            b11 = hessian(u_next,c1,c1,e)
            b12 = hessian(u_next,c1,c2,e)
            b22 = hessian(u_next,c2,c2,e)

            beta2 = 1
            beta1 = -b22/(2*b12)
            c_star = beta1*c1 + beta2*c2
            c_star = c_star / np.linalg.norm(c_star)

            to_explore.append((cs.vertcat(deepcopy(z_star),deepcopy(E0)),deepcopy(c1),-np.sign(deepcopy(sign))))
            to_explore.append((cs.vertcat(deepcopy(z_star),deepcopy(E0)),deepcopy(c_star),-np.sign(deepcopy(sign))))
            break
        

        if i>5 and p_list[i-2].T @ p_list[i-1] < 0:
            d = 0.5 * d
            precision += 1

        


fig = plt.figure()
ax = fig.add_subplot(1,1,1,projection="3d")
for i in range(len(connected_component)):
    branch = np.array(connected_component[i])
    plt.plot(branch[:,1],branch[:,2],branch[:,3])
plt.show()
