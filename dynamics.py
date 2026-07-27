import casadi as cs
import hyperparameters as params

xs = cs.SX.sym("xs", 4)
xf = cs.SX.sym("xf", 6)
dt = cs.SX.sym("dt")

l = cs.sqrt(xs[0] ** 2 + xs[1] ** 2)
Es = 0.5 * (xs[2] ** 2 + xs[3] ** 2) + xs[1] + 0.5 * (l - 1) ** 2
Ef = 0.5 * (xf[3] ** 2 + xf[4] ** 2) + xf[1]
dxs = cs.vertcat(xs[2:], params.K * (1 - l) * xs[0] / l, params.K * (1 - l) * xs[1] / l - 1)
dxf = cs.vertcat(xf[3:], 0, -1, -(params.W**2) * xf[2])

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
traj_f = RK4f.fold(params.N_F)
traj_s = RK4s.fold(params.N_S)

s2f = cs.vertcat(
    xs[:2], cs.atan(-xs[0] / xs[1]), xs[2:], (xs[0] * xs[3] - xs[1] * xs[2]) / l
)
f2s = cs.vertcat(-cs.sin(xf[2]), cs.cos(xf[2]), xf[3:5])
stance_to_flight = cs.Function("stance_to_flight", [xs], [s2f])
flight_to_stance = cs.Function("flight_to_stance", [xf], [f2s])
