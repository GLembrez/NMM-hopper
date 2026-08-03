import casadi as cs
import dynamics

N_NEWTON = 10
STEP_SIZE = 1

x_TD_est = cs.SX.sym("x_TD_est",5)
x_LO_est = cs.SX.sym("x_LO_est",4)
x_apex_est = cs.SX.sym("x_apex_est",5)

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

R_apex = cs.vertcat(x_apex[1:] - x_apex_est,x_apex[4], cs.cos(x_TD[2]) - x_TD[1], x_LO[0] ** 2 + x_LO[1] ** 2 - 1,)
R_LO = cs.vertcat(x_LO - x_LO_est, cs.cos(x_TD[2]) - x_TD[1], x_LO[0] ** 2 + x_LO[1] ** 2 - 1)
R_TD = cs.vertcat(x_TD[1:] - x_TD_est, cs.cos(x_TD[2]) - x_TD[1])
z_apex = cs.vertcat(x_apex_est,t)
z_TD = cs.vertcat(x_TD_est,t[0])
z_LO = cs.vertcat(x_LO_est,t[:2])
J_apex = cs.jacobian(R_apex,z_apex)
J_TD = cs.jacobian(R_TD,z_TD)
J_LO = cs.jacobian(R_LO,z_LO)
newton_apex = z_apex - cs.inv(J_apex) @ R_apex
newton_LO = z_LO - cs.inv(J_LO) @ R_LO
newton_TD = z_TD - cs.inv(J_TD) @ R_TD
jac_newton_apex = cs.jacobian(newton_apex[1:6],x_init[1:]) # TODO fold newton
jac_newton_LO = cs.jacobian(newton_LO[1:6],x_init[1:])
jac_newton_TD = cs.jacobian(newton_TD[1:6],x_init[1:])
compute_apex_ = cs.Function("apex",[z_apex,x_init,xi],[newton_apex])
compute_apex = compute_apex_.fold(N_NEWTON)
compute_LO_ = cs.Function("LO",[z_LO,x_init,xi],[newton_LO])
compute_LO = compute_LO_.fold(N_NEWTON)
compute_TD_ = cs.Function("TD",[z_TD,x_init,xi],[newton_TD])
compute_TD = compute_TD_.fold(N_NEWTON)


apex_star = compute_apex(cs.vertcat(x_TD[1:],t),x_init,xi)
TD_star = compute_TD(cs.vertcat(x_TD[1:],t[0]),x_init,xi)
LO_star = compute_LO(cs.vertcat(x_LO,t[:2]),x_init,xi)

root = cs.vertcat(
    apex_star[1:6] - x_init[1:],
    x_init[0],
    x_init[4],
    dynamics.energy_flight(x_init) - E,
    cs.cos(TD_star[2]) - TD_star[1],
    LO_star[0] ** 2 + LO_star[1] ** 2 - 1,
)

J_star = cs.horzcat(jac_newton_apex,dynamics.ff(cs.vertcat(0,x_apex_est),xi)[1:])
compute_J_star = cs.Function("compute_J_star",[u,x_apex_est,x_LO_est,x_TD_est],[J_star])

J_root = cs.jacobian(root, z)
J_ext = cs.jacobian(root, u)
z_newton = z - STEP_SIZE*cs.inv(J_root) @ root

b = du1.T @ cs.jtimes(J_ext, u, du2).T @ phi

newton_step = cs.Function("newton_step", [z, E,x_apex_est,x_LO_est,x_TD_est], [z_newton])
newton = newton_step.fold(N_NEWTON)
residual = cs.Function("residual", [u,x_apex_est,x_LO_est,x_TD_est], [root])
compute_J = cs.Function("compute_J", [u,x_apex_est,x_LO_est,x_TD_est], [J_ext])
hessian = cs.Function("Hessian", [u, du1, du2, phi,x_apex_est,x_LO_est,x_TD_est], [b])


