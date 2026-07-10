import casadi as cs
import numpy as np
import hyperparameters as params
import dynamics
from scipy.linalg import eig

flight_integrate = dynamics.RK4f.fold(params.N_F // 2 + 1)
stance_integrate = dynamics.RK4s.fold(params.N_S)

flight_accum = dynamics.RK4f.mapaccum(params.N_F // 2 + 1)
stance_accum = dynamics.RK4s.mapaccum(params.N_S)

K = cs.SX.sym("K")
W = cs.SX.sym("W")

var = cs.SX.sym("var", 19)
x_init = cs.SX.sym("x_init", 6)
residual = cs.vertcat(
    var[:6] - flight_integrate(x_init, 0.0, var[16], W),
    var[6:10] - stance_integrate(dynamics.flight_to_stance(var[:6]), 0.0, var[17], K),
    var[10:16]
    - flight_integrate(dynamics.stance_to_flight(var[6:10]), 0.0, var[18], W),
    cs.cos(var[2]) - var[1],
    var[6] ** 2 + var[7] ** 2 - 1,
    var[14],
)
newton = var - cs.inv(cs.jacobian(residual, var)) @ residual
newton_jac = cs.jacobian(newton, x_init)
step_newton = cs.Function("step_newton", [var, x_init, K, W], [newton])
residual_newton = cs.Function("residual_newton", [var, x_init, K, W], [residual])
monodromy = cs.Function(
    "monodromy",
    [var, x_init, K, W],
    [newton_jac[[11, 12, 13, 15], [1, 2, 3, 5]], newton_jac[14, [1, 2, 3, 5]]],
)
full_newton = step_newton.fold(5)

traj_s = stance_accum(dynamics.flight_to_stance(var[:6]), 0.0, var[17], K)
traj_f = cs.horzcat(
    flight_accum(dynamics.stance_to_flight(var[6:10]), 0.0, var[18], W),
    flight_accum(var[10:16], 0.0, var[16], W)[:, 1:],
)
trajectory = cs.Function("trajectory", [var, K, W], [traj_f, traj_s, var[18], var[17]])


def compute_multipliers(traj_compact, apex, dir_old, K, W):
    var = full_newton(traj_compact, apex, K, W)
    M, grad_P = monodromy(var, apex, K, W)
    f = dynamics.ff(apex, 0.0, W)[[1, 2, 3, 5]]
    D = M - np.eye(4) - (f @ grad_P)/(grad_P @ f) @ M
    mul, dir = eig(D)
    mul = np.abs(mul)
    print(grad_P,f)
    print(M)
    idx_sorted = np.argsort(np.abs(mul))
    dir1, dir2 = dir[:, idx_sorted[0]], dir[:, idx_sorted[1]]
    if np.linalg.norm(np.abs(dir1) - np.abs(dir_old)) < np.linalg.norm(
        np.abs(dir2) - np.abs(dir_old)
    ):
        dir_continuation = dir1
        dir_bifurcation = dir2
    else:
        dir_continuation = dir2
        dir_bifurcation = dir1
    return dir_continuation, dir_bifurcation, mul[idx_sorted[1]]
