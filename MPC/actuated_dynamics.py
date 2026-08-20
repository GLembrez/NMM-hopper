import casadi as cs


def build(solver):

    W = solver.W
    K = solver.K
    N = solver.N

    xs = cs.SX.sym("xs", 4)
    xf = cs.SX.sym("xf", 6)
    u = cs.SX.sym("u")
    dt = cs.SX.sym("dt")

    l = cs.sqrt(xs[0] ** 2 + xs[1] ** 2)
    dxs = cs.vertcat(
        xs[2:], (K * (1 - l) + u) * xs[0] / l, (K * (1 - l) + u) * xs[1] / l - 1
    )
    dxf = cs.vertcat(xf[3:], 0, -1, -(W**2) * xf[2] + u)

    ff = cs.Function("ff", [xf, u], [dxf])
    fs = cs.Function("fs", [xs, u], [dxs])

    k1s = fs(xs, u)
    k1f = ff(xf, u)
    k2s = fs(xs + dt / 2 * k1s, u)
    k2f = ff(xf + dt / 2 * k1f, u)
    k3s = fs(xs + dt / 2 * k2s, u)
    k3f = ff(xf + dt / 2 * k2f, u)
    k4s = fs(xs + dt * k3s, u)
    k4f = ff(xf + dt * k3f, u)
    xs_ = xs + dt / 6 * (k1s + 2 * k2s + 2 * k3s + k4s)
    xf_ = xf + dt / 6 * (k1f + 2 * k2f + 2 * k3f + k4f)

    s2f = cs.vertcat(
        xs[:2], cs.atan(-xs[0] / xs[1]), xs[2:], (xs[0] * xs[3] - xs[1] * xs[2]) / l
    )
    f2s = cs.vertcat(-cs.sin(xf[2]), cs.cos(xf[2]), xf[3:5])

    stance_to_flight = cs.Function("stance_to_flight", [xs], [s2f])
    flight_to_stance = cs.Function("flight_to_stance", [xf], [f2s])
    RK4s = cs.Function("RK4s", [xs, dt, u], [xs_])
    RK4f = cs.Function("RK4f", [xf, dt, u], [xf_])

    traj_f = RK4f.fold(N)
    traj_s = RK4s.fold(N)
    traj_f_list = RK4f.mapaccum(N)
    traj_s_list = RK4s.mapaccum(N)

    solver.stance_to_flight = stance_to_flight
    solver.flight_to_stance = flight_to_stance
    solver.traj_f = traj_f
    solver.traj_s = traj_s
    solver.traj_f_list = traj_f_list
    solver.traj_s_list = traj_s_list
    solver.RK4s = RK4s
    solver.RK4f = RK4f
