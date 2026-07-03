import casadi as cs
import hyperparameters as params
import dynamics
import numpy as np

K = cs.SX.sym("K")
W = cs.SX.sym("W")

xs_LO = cs.SX.sym("xs_LO", 4)
xs_TD = cs.SX.sym("xs_TD", 4)
xf_LO = cs.SX.sym("xf_LO", 6)
xf_TD = cs.SX.sym("xf_TD", 6)
dt_s = cs.SX.sym("dt_s")
dt_f = cs.SX.sym("dt_f")

var_s = cs.vertcat(xs_LO, dt_s)
var_f = cs.vertcat(xf_TD, dt_f)
xf_TD_est = dynamics.flight_integrate(xf_LO, 0.0, dt_f, W)
xs_LO_est = dynamics.stance_integrate(xs_TD, 0.0, dt_s, K)
residual_f = cs.vertcat(xf_TD_est - xf_TD, xf_TD_est[1] - cs.cos(xf_TD_est[2]))
residual_s = cs.vertcat(xs_LO_est - xs_LO, xs_LO_est[0] ** 2 + xs_LO_est[1] ** 2 - 1)
newton_s = var_s - cs.inv(cs.jacobian(residual_s, var_s)) @ residual_s
newton_f = var_f - cs.inv(cs.jacobian(residual_f, var_f)) @ residual_f

eval_newton_s = cs.Function(
    "eval_newton_s", [xs_TD, var_s, K], [newton_s, cs.jacobian(newton_s[:4], xs_TD)]
)
eval_newton_f = cs.Function(
    "eval_newton_f", [xf_LO, var_f, W], [newton_f, cs.jacobian(newton_f[:6], xf_LO)]
)


def compute_floquet(branch, idx, K, W):
    var_s = cs.vertcat(branch["traj"][idx][[0, 1, 3, 4], -1], branch["dt"][idx][1])
    var_f = cs.vertcat(branch["traj"][idx][:, params.N - 1], branch["dt"][idx][0])
    for _ in range(10):
        var_s, J_s = eval_newton_s(
            branch["traj"][idx][[0, 1, 3, 4], params.N], var_s, K
        )
        var_f, J_f = eval_newton_f(branch["traj"][idx][:, 0], var_f, W)
    J_f2s = dynamics.Jf2s(branch["traj"][idx][:, params.N - 1])
    monodromy_matrix = J_s @ J_f2s @ J_f
    return monodromy_matrix[:, [0, 1, 3, 4]]
