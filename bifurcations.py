import casadi as cs
import dynamics

N_NEWTON = 3
STEP_SIZE = 1

x_init = cs.SX.sym("x_init", 6)
t = cs.SX.sym("t", 3)
xi = cs.SX.sym("xi")
E = cs.SX.sym("E")
du1 = cs.SX.sym("du1", 11)
du2 = cs.SX.sym("du2", 11)
phi = cs.SX.sym("phi", 10)

z = cs.vertcat(x_init, t, xi)
u = cs.vertcat(z, E)

x_TD = dynamics.traj_f(x_init, t[0], xi)
x_LO = dynamics.traj_s(dynamics.flight_to_stance(x_TD), t[1], xi)
x_apex = dynamics.traj_f(dynamics.stance_to_flight(x_LO), t[2], xi)

root = cs.vertcat(
    x_apex[1:] - x_init[1:],
    x_init[0],
    x_init[4],
    dynamics.energy_flight(x_init) - E,
    cs.cos(x_TD[2]) - x_TD[1],
    x_LO[0] ** 2 + x_LO[1] ** 2 - 1,
)

root_rev = cs.vertcat(
    x_apex[[1, 4]] - x_init[[1, 4]],
    x_apex[[2, 3, 5]] + x_init[[2, 3, 5]],
    x_init[0],
    x_init[4],
    dynamics.energy_flight(x_init) - E,
    cs.cos(x_TD[2]) - x_TD[1],
    x_LO[0] ** 2 + x_LO[1] ** 2 - 1,
)

J_root = cs.jacobian(root, z)
J_ext = cs.jacobian(root, u)
J_root_rev = cs.jacobian(root_rev, z)
J_ext_rev = cs.jacobian(root_rev, u)
z_newton = z - STEP_SIZE*cs.inv(J_root) @ root
z_newton_rev = z - cs.inv(J_root_rev) @ root_rev

b = du1.T @ cs.jtimes(J_ext, u, du2).T @ phi

newton_step = cs.Function("newton_step", [z, E], [z_newton])
newton_step_rev = cs.Function("newton_step_rev", [z, E], [z_newton_rev])
newton = newton_step.fold(N_NEWTON)
newton_rev = newton_step_rev.fold(N_NEWTON)
residual = cs.Function("residual", [u], [root])
residual_rev = cs.Function("residual_rev", [u], [root_rev])
compute_J = cs.Function("compute_J", [u], [J_ext])
compute_J_rev = cs.Function("compute_J_rev", [u], [J_ext_rev])
hessian = cs.Function("Hessian", [u, du1, du2, phi], [b])


