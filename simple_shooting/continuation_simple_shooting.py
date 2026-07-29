import casadi as cs
import numpy as np
from scipy.linalg import null_space
from time import time 

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

z0 = np.array([0.0,1.124,0.,0.,0.,0.,0.1,0.01,0.1,0.0])
E = z0[1]
d = -0.005
p_list = []
u_list = []
precision = 0

start = time()
for i in range(1000):
    z_star = newton_step(z0,E)
    u_list.append(cs.vertcat(z_star,E))
    R_eval = compute_R(cs.vertcat(z_star,z_star[1]))
    p = null_space(R_eval)
    if i>5 and p_list[i-2].T @ p_list[i-1] < 0:
        d = 0.5 * d
        precision += 1
    if np.linalg.det(cs.vertcat(R_eval,p.T)) < 0:
        p = -p
    # else:
    u_next = cs.vertcat(z_star,E) +  d * p
    if precision == 10:
        print(i)
        break
    z0 = u_next[:10]
    E = u_next[10]
    p_list.append(p)
end = time()
print(end-start,E)

c1 = -p 
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

d = -0.005
precision = 0
u_next = cs.vertcat(z_star,E) +  d * c_star
z0 = u_next[:10]
E = u_next[10]
start = time()
for i in range(100):
    print(z0[3])
    z_star = newton_step(z0,E)
    u_list.append(cs.vertcat(z_star,E))
    R_eval = compute_R(cs.vertcat(z_star,z_star[1]))
    p = null_space(R_eval)
    if i>5 and p_list[i-2].T @ p_list[i-1] < 0:
        d = 0.5 * d
        precision += 1
    if np.linalg.det(cs.vertcat(R_eval,p.T)) < 0:
        p = -p
    # else:
    u_next = cs.vertcat(z_star,E) +  d * p
    if precision == 10:
        print(i)
        break
    z0 = u_next[:10]
    E = u_next[10]
    p_list.append(p)
end = time()
print(end-start,z0)
