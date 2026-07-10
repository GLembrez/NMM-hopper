import casadi as cs
import numpy as np
from scipy.linalg import eig

np.set_printoptions(precision=3)

N_F = 25
N_S = 50
N_NEWTON = 10
K = 40
W = cs.sqrt(10)

xs = cs.SX.sym("xs", 4)
xf = cs.SX.sym("xf", 6)
epsilon = cs.SX.sym('epsilon')
u = cs.SX.sym("u")
dt = cs.SX.sym("dt")

l = cs.sqrt(xs[0] ** 2 + xs[1] ** 2)
energy_stance = 0.5*(xs[2]**2 + xs[3]**2) + xs[1] + 0.5*(l-1)**2
energy_flight = 0.5*(xf[3]**2 + xf[4]**2) + xf[1]
dxs = cs.vertcat(
    xs[2:], K * (1 - l)  * xs[0] / l, K * (1 - l)  * xs[1] / l - 1
) + epsilon * cs.jacobian(energy_stance,xs).T
dxf = cs.vertcat(xf[3:], 0, -1, -(W**2) * xf[2] ) + epsilon * cs.jacobian(energy_flight,xf).T

ff = cs.Function("ff", [xf, epsilon], [dxf])
fs = cs.Function("fs", [xs, epsilon], [dxs])

k1s = fs(xs, epsilon)
k1f = ff(xf, epsilon)
k2s = fs(xs + dt / 2 * k1s, epsilon)
k2f = ff(xf + dt / 2 * k1f, epsilon)
k3s = fs(xs + dt / 2 * k2s, epsilon)
k3f = ff(xf + dt / 2 * k2f, epsilon)
k4s = fs(xs + dt * k3s, epsilon)
k4f = ff(xf + dt * k3f, epsilon)
xs_ = xs + dt / 6 * (k1s + 2 * k2s + 2 * k3s + k4s)
xf_ = xf + dt / 6 * (k1f + 2 * k2f + 2 * k3f + k4f)

RK4s = cs.Function("RK4s", [xs, dt, epsilon], [xs_])
RK4f = cs.Function("RK4f", [xf, dt, epsilon], [xf_])
traj_f = RK4f.fold(N_F)
traj_s = RK4s.fold(N_S)

s2f = cs.vertcat(xs[:2], cs.atan(-xs[0] / xs[1]), xs[2:], xs[0] * xs[3] - xs[1] * xs[2])
f2s = cs.vertcat(-cs.sin(xf[2]), cs.cos(xf[2]), xf[3:5])
stance_to_flight = cs.Function('stance_to_flight',[xs],[s2f])
flight_to_stance = cs.Function('flight_to_stance',[xf],[f2s])

dates = cs.SX.sym("dates",2)
x_star = cs.SX.sym("x_star",6)
energy = cs.SX.sym("energy")
z = cs.vertcat(x_star,dates,epsilon,energy)


x_TD = traj_f(x_star,dates[0],epsilon)
x_LO = traj_s(flight_to_stance(x_TD),dates[1],epsilon)
x_APEX = traj_f(stance_to_flight(x_LO),dates[0],epsilon)
R = cs.vertcat(
    x_star[1:] - x_APEX[1:],
    x_star[0],
    x_star[4],
    energy - 0.5*(x_star[3]**2 + x_star[4]**2) - x_star[1],
    cs.cos(x_TD[2]) - x_TD[1],
    x_LO[0]**2 + x_LO[1]**2 - 1 
)
jac_R = cs.jacobian(R,z)

newton = z - cs.inv(cs.jacobian(R,z)) @ R
residual = cs.Function("residual",[z],[R])
residual_jacobian = cs.Function("jac_residual",[z],[jac_R])
newton_step = cs.Function("newton_step",[z],[newton])
newton_full = newton_step.fold(N_NEWTON)

z_eval = [0.,1.255,0.,0.,0.,0.,0.02857,0.011,0.,1.255]
z_star = newton_full(z_eval)
J = residual_jacobian(z_star)
mul,dir = eig(J)
idx_sorted = np.argsort(np.abs(mul))
p = np.real(dir[:,idx_sorted[0]])

if np.linalg.det(cs.vertcat(J,p.T)) < 0:
    p = -p
    
print(p)

# opti = cs.Opti()
# opti.solver("ipopt", {"print_time": 0, "ipopt.print_level": 0, "ipopt.tol": 1e-12})
# var = opti.variable(10)
# opti.subject_to(newton_full(var).T @ newton_full(var)  < 1e-3)
# opti.set_initial(var,z_star)
# opti.solve()
# print(opti.value(var))