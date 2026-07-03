import casadi as cs
import hyperparameters as params

K = cs.SX.sym("K")
W = cs.SX.sym("W")

xs = cs.SX.sym("xs", 4)
xf = cs.SX.sym("xf", 6)
u = cs.SX.sym("u", 2)
dt = cs.SX.sym("dt")

l = cs.sqrt(xs[0] ** 2 + xs[1] ** 2)
dxs = cs.vertcat(
    xs[2:], (K * (1 - l) + u[0]) * xs[0] / l, (K * (1 - l) + u[0]) * xs[1] / l - 1
)
dxf = cs.vertcat(xf[3:], 0, -1, -(W**2) * xf[2] + u[1])

ff = cs.Function("ff", [xf, u, W], [dxf])
fs = cs.Function("fs", [xs, u, K], [dxs])

k1s = fs(xs, u, K)
k1f = ff(xf, u, W)
k2s = fs(xs + dt / 2 * k1s, u, K)
k2f = ff(xf + dt / 2 * k1f, u, W)
k3s = fs(xs + dt / 2 * k2s, u, K)
k3f = ff(xf + dt / 2 * k2f, u, W)
k4s = fs(xs + dt * k3s, u, K)
k4f = ff(xf + dt * k3f, u, W)
xs_ = xs + dt / 6 * (k1s + 2 * k2s + 2 * k3s + k4s)
xf_ = xf + dt / 6 * (k1f + 2 * k2f + 2 * k3f + k4f)

RK4s = cs.Function("RK4s", [xs, u, dt, K], [xs_])
RK4f = cs.Function("RK4f", [xf, u, dt, W], [xf_])
flight_integrate = RK4f.fold(params.N)
stance_integrate = RK4s.fold(params.N)

s2f = cs.vertcat(xs[:2], cs.atan(-xs[0] / xs[1]), xs[2:], xs[0] * xs[3] - xs[1] * xs[2])
f2s = cs.vertcat(-cs.sin(xf[2]), cs.cos(xf[2]), xf[3:5])
Js2f = cs.Function("Js2f", [xs], [cs.jacobian(s2f, xs)])
Jf2s = cs.Function("Jf2s", [xf], [cs.jacobian(f2s, xf)])
