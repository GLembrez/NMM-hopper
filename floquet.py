import casadi as cs
import hyperparameters as params
import dynamics


flight_integrate = dynamics.RK4f.fold(params.N//2)
stance_integrate = dynamics.RK4s.fold(params.N)

K = cs.SX.sym("K")
W = cs.SX.sym("W")

var = cs.SX.sym("var", 19)
x_init = cs.SX.sym("x_init", 6)
residual = cs.vertcat(
    var[:6] - dynamics.flight_integrate(x_init, var[16], 0, W),
    var[6:10] - dynamics.stance_integrate(dynamics.flight_to_stance(var[:6]), var[17], 0, K),
    var[10:16] - dynamics.flight_integrate(dynamics.stance_to_flight(var[6:10]), var[18], 0, W),
    cs.cos(var[2]) - var[1],
    var[6] ** 2 + var[7] ** 2,
    var[14],
)
newton = var - cs.jacobian(residual, var) @ var
newton_jac = cs.jacobian(newton, x_init)

monodromy = cs.Function('monodromy',[x_init,var,K,W],[newton_jac[[11,12,13,15],[1,2,3,5]]])


def compute_floquet(branch, idx, K, W):
    x_init = branch["traj"][idx][:,params.N//2]
    var = cs.vertcat(
        branch["traj"][idx][:,params.N//2],
        branch["traj"][idx][:,params.N+params.N//2],
        branch["traj"][idx][:,2*params.N-1],
        branch["dt_f"][idx]/2,
        branch["dt_s"][idx],
        branch["dt_f"][idx]/2
    )
    return monodromy(x_init,var,K,W)
