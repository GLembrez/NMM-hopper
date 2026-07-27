import casadi as cs
import dynamics
from scipy.linalg import eig

reverse_matrix = cs.DM([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,-1]])

init = cs.SX.sym("init", 5)
apex = cs.SX.sym("apex", 5)
times = cs.SX.sym("times", 2)
z = cs.vertcat(apex, times)

x_TD = dynamics.traj_f(cs.vertcat(0.0,init), times[0])
x_LO = dynamics.traj_s(dynamics.flight_to_stance(x_TD), times[1])
x_a = dynamics.traj_f(dynamics.stance_to_flight(x_LO), times[0])

R = cs.vertcat(
    x_a[1:] - apex,
    cs.cos(x_TD[2]) - x_TD[1],
    x_LO[0] ** 2 + x_LO[1] ** 2 - 1,
)
jac_R = cs.jacobian(R, z)
newton = z - cs.inv(jac_R) @ R
jac_newton = cs.jacobian(newton[0, 1, 2, 4] - init[0, 1, 2, 4], init[0, 1, 2, 4])
jac_newton_reverse = cs.jacobian(newton[0, 1, 2, 4] - reverse_matrix @ init[0, 1, 2, 4],init[0, 1, 2, 4])

eval_newton = cs.Function("eval_newton", [z, init], [newton])
full_newton = eval_newton.fold(10)
monodromy = cs.Function(
    "monodromy", [init, apex, times], [jac_newton]
)
monodromy_reverse = cs.Function(
    "monodromy_reverse", [init, apex, times], [jac_newton_reverse]
)
