import casadi as cs
import numpy as np

np.set_printoptions(precision=9,suppress=True)

N_F = 25
N_S = 50
K = 40
W = cs.sqrt(10)


xs = cs.SX.sym("xs", 4)
xf = cs.SX.sym("xf", 6)
dt = cs.SX.sym("dt")

l = cs.sqrt(xs[0] ** 2 + xs[1] ** 2)
Es = 0.5 * (xs[2] ** 2 + xs[3] ** 2) + xs[1] + 0.5 * (l - 1) ** 2
Ef = 0.5 * (xf[3] ** 2 + xf[4] ** 2) + xf[1]
dxs = cs.vertcat(xs[2:], K * (1 - l) * xs[0] / l, K * (1 - l) * xs[1] / l - 1) #+ xi * cs.gradient(Es,xs)
dxf = cs.vertcat(xf[3:], 0, -1, -(W**2) * xf[2]) #+ xi * cs.gradient(Ef,xf)

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

z = cs.SX.sym('z',7)
init = cs.SX.sym("init", 5)

x_TD = traj_f(cs.vertcat(0.0,init), z[5])
x_LO = traj_s(flight_to_stance(x_TD), z[6])
x_a = traj_f(stance_to_flight(x_LO), z[5])

R = cs.vertcat(
    x_a[1:] - z[:5],
    cs.cos(x_TD[2]) - x_TD[1],
    x_LO[0] ** 2 + x_LO[1] ** 2 - 1,
)
jac_R = cs.jacobian(R, z)
z_star = z - cs.inv(jac_R) @ R


x_TD_star = traj_f(cs.vertcat(0.0,init), z_star[5])
x_LO_star = traj_s(flight_to_stance(x_TD_star), z_star[6])
x_a_star = traj_f(stance_to_flight(x_LO_star), z_star[5])

R_star = cs.vertcat(
    x_a_star[1:] - z_star[:5],
    cs.cos(x_TD_star[2]) - x_TD_star[1],
    x_LO_star[0] ** 2 + x_LO_star[1] ** 2 - 1,
)

monodromy = cs.Function("monodromy", [init,z], [cs.jacobian(z_star[0,1,2,4],init[0,1,2,4]),z_star])



z0 = [1.12386 , 0.0, 0.0, 0.0, 0.0, 0.02, 0.012]
M,z_star = monodromy([1.12386 , 0.0, 0.0, 0.0, 0.0],z0) #1.12386
print(z_star)
val,vec = np.linalg.eig(M)
idx_sorted = np.argsort(np.abs(val))
print(val[idx_sorted[:5]])
print(vec[:,idx_sorted[:3]].real)
