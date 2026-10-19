import numpy as np
import casadi as cs


def bounded(expression, lower=0, upper=0):
    # converts expression in three equally sized vectors inf < expression < upper 
    # by default, expression = 0
    expression = cs.vec(expression)
    ones = cs.DM.ones(expression.numel())
    return expression, lower*ones, upper*ones

def make_problem(layout,target_LUT,dynamics,rho_regul=1e-6,rho_slack=1e3):
    w = cs.MX.sym("w", layout.nw)
    p = cs.MX.sym("p", layout.dim[layout.modes[0]]+1)
    y,v = layout.split(w)
    x0,T = p[:-1],p[-1]

    effort = []
    for k, (_, mode, _) in enumerate(layout.stages):
        if layout.edge[k] == "flow":
            _, tau, _ = layout.state_parts(y, k)
            h = tau/(layout.N[mode]-1)
            effort.append(cs.sqrt(2*h)*v[k][0])

    E,phi,s = cs.vertsplit(y[-1][-layout.dim_terminal_aux:], [0,1,2,layout.dim_terminal_aux])
    s_plus, s_minus = cs.vertsplit(s, [0, layout.dim_terminal, 2*layout.dim_terminal])
    q = cs.vertcat(*effort)
    cost = 0.5*cs.dot(q,q) + rho_slack * cs.sum1(s)
    grad = cs.gradient(cost, w)
    Q = cs.jacobian(q, w)
    H = Q.T @ Q + rho_regul * cs.DM.eye(layout.nw)

    residuals = []          # all constraint expressions OCP-ordered: [dynamics gap_k \\ local constraints_k]
    lower,upper = [],[]     # corresponding bounds : lower <= residuals <= upper
    ng = []                 # ng[k]: number non-dynamics constraint at stage k
    for k, (idx_m,m,idx_n) in enumerate(layout.stages):
        x,tau,t = layout.state_parts(y,k)


        if k<layout.N_ocp: # non terminal stage
            if layout.edge[k] == "flow":
                h = tau / (layout.N[m] - 1)
                y_est = cs.vertcat(dynamics.rk4[m](x,h,v[k][0]),tau,t)
                if k == layout.N_ocp-1:
                    # propagate the augmented capture state using augmented control v_(N-1)
                    y_est = cs.vertcat(y_est, v[k][1:])
            else: # reset stage
                y_est = cs.vertcat(dynamics.convert[m](x),v[k][0], tau+t)
            gap,lb,ub = bounded(y_est-y[k+1])
            residuals.append(gap)
            lower.append(lb)
            upper.append(ub)

        local = [] # local non-dynamics constraints
        if k == 0:
            local.append(bounded(cs.vertcat(x-x0,t)))
        if layout.edge[k] == "reset":
            local.append(bounded(dynamics.guard[m](x)))
            local.append(bounded(dynamics.guard_rate[m](x),1e-3, cs.inf))
        if idx_m == len(layout.modes)-1 and idx_n > 0:
            # every nodes of the last mode except the first one
            local.append(bounded(dynamics.guard[m](x), -cs.inf, -1e-3))
        if k == layout.N_ocp: # terminal 
            target = target_LUT(cs.vertcat(E,phi))
            x_target = target[1:-1] if m == "f" else target[:-1]
            x_capture = x[1:] if m == "f" else x
            local.append(bounded(x_target-x_capture-s_plus+s_minus))
            local.append(bounded(t+tau-T))

        ng.append(sum(expression.numel() for expression, _, _ in local))
        for expression, lb, ub in local:
            residuals.append(expression)
            lower.append(lb)
            upper.append(ub)

    residual = cs.vertcat(*residuals)
    A = cs.jacobian(residual, w)
    linearize = cs.Function(
        "linearize",
        [w,p],
        [grad,H,A,residual],
        {"jit": True, "compiler": "shell"}
    )
    values = cs.Function(
        "values",
        [w,p],
        [cost,residual]
    )
    structure = {"h": H.sparsity(), "a": A.sparsity()}
    return linearize, values, structure, cs.vertcat(*lower), cs.vertcat(*upper), ng

def make_bounds(layout, x_bounds, u_bounds, t_bounds, E_bounds, phi_bounds):
    lower = np.full(layout.nw, -np.inf)
    upper = np.full(layout.nw, np.inf)
    for k, (idx_m, m, _) in enumerate(layout.stages):
        d = layout.dim[m]
        y_s,v_s = layout.x_slice[k], layout.u_slice[k]
        for idx_b,bounds in x_bounds.get(m, {}).items():
            lower[y_s.start+idx_b], upper[y_s.start+idx_b] = bounds 
        lower[y_s.start+d],upper[y_s.start+d] = t_bounds[m]
        if layout.edge[k] == "flow":
            lower[v_s.start], upper[v_s.start] = u_bounds[m]
        elif layout.edge[k] == "reset":
            # tmin < t(k+1) < tmax
            lower[v_s], upper[v_s] = t_bounds[layout.modes[idx_m + 1]]

    for aux in (layout.u_t_slice,layout.x_t_slice):
        lower[aux.start],upper[aux.start] = E_bounds
        lower[aux.start+1],upper[aux.start+1] = phi_bounds
        lower[aux.start+2:aux.stop] = 0.0

    return lower,upper

def make_delta(layout):
    # constructs the trust-region radius for every entry in delta w
    delta = np.full(layout.nw, np.inf)
    for k, (_,m,_) in enumerate(layout.stages):
        d = layout.dim[m]
        y_s,v_s = layout.x_slice[k], layout.u_slice[k]
        delta[y_s.start:y_s.start + d] = 0.2    # physical state increment bound
        delta[y_s.start+d] = 0.1                # mode duration increment bound
        if layout.edge[k] == "flow":
            delta[v_s.start] = 0.5              # control signal increment bound
        elif layout.edge[k] == "reset":
            delta[v_s] = 0.1                    # mode duration increment bound
    for aux in (layout.u_t_slice,layout.x_t_slice):
        delta[aux.start] = 0.1                  # E increment bound
        delta[aux.start+1] = 0.2                # phi increment bound
    return delta

