import casadi as cs

A1 = cs.DM([[0,1,0,0,0,0],[0,0,0,0,1,0]])
A2 = cs.DM([[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,0,1]])
A3 = cs.DM([[1,0,0,0,0,0]])

def build_casadi_functions(dynamics_functions, PERIODIC):

    stance_to_flight,flight_to_stance,traj_f,traj_s,energy_flight = dynamics_functions

    Ap = A1
    Aap = -A2 if PERIODIC else A2
    Anp = A3

    x_init = cs.SX.sym("x_init", 6)
    t = cs.SX.sym("t", 3)
    E = cs.SX.sym("E")
    du1 = cs.SX.sym("du1", 11)
    du2 = cs.SX.sym("du2", 11)
    phi = cs.SX.sym("phi", 10)
    xi = cs.SX.sym("xi")

    z = cs.vertcat(x_init, t, xi)
    u = cs.vertcat(z, E)

    x_TD = traj_f(x_init, t[0], xi)
    x_LO = traj_s(flight_to_stance(x_TD), t[1], xi)
    x_apex = traj_f(stance_to_flight(x_LO), t[2], xi)

    root = cs.vertcat(
        Ap @ x_apex - A1 @ x_init,
        Aap @ x_apex + A2 @ x_init,
        Anp @ x_init,
        x_init[4],
        energy_flight(x_init) - E,
        cs.cos(x_TD[2]) - x_TD[1],
        x_LO[0] ** 2 + x_LO[1] ** 2 - 1,
    )

    J_root = cs.jacobian(root, z)
    J_ext = cs.jacobian(root, u)
    z_newton = z - cs.inv(J_root) @ root

    b = du1.T @ cs.jtimes(J_ext, u, du2).T @ phi

    compute_J = cs.Function("compute_J", [u], [J_ext])
    compute_Jz = cs.Function("compute_Jz",[z,E],[J_root])
    residual = cs.Function("residual", [u], [root])
    hessian = cs.Function("Hessian", [u, du1, du2, phi], [b])

    return compute_J,compute_Jz, residual, hessian



