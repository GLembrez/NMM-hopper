import casadi as cs
from scipy.optimize import brentq

def RK4(f,x,h,u):
    k1 = f(x,u)
    k2 = f(x+0.5*h*k1,u)
    k3 = f(x+0.5*h*k2,u)
    k4 = f(x+h*k3,u)
    x_next = x + h/6 * (k1 + 2*k2 + 2*k3 +k4)
    return x_next

def build(solver):

    W = solver.W
    K = solver.K

    xs = cs.SX.sym("xs", 4)
    xf = cs.SX.sym("xf", 6)
    u = cs.SX.sym("u")
    h = cs.SX.sym("h")

    l = cs.sqrt(xs[0] ** 2 + xs[1] ** 2)
    dxs = cs.vertcat(
        xs[2:], 
        (K * (1 - l) + u) * xs[0] / l, 
        (K * (1 - l) + u) * xs[1] / l - 1
    )
    dxf = cs.vertcat(
        xf[3:], 
        0, 
        -1, 
        -(W**2) * xf[2] + u
    )
    xfs = cs.vertcat(
        xs[:2],
        cs.atan2(-xs[0],xs[1]),
        xs[2:],
        (xs[0] * xs[3] - xs[1] * xs[2]) / l**2
    )
    xsf = cs.vertcat(
        -cs.sin(xf[2]),
        xf[[1,3,4]]
    )
    e_TD = cs.cos(xf[2]) - xf[1]
    e_LO = l - 1

    f =  {
        "f": cs.Function("ff", [xf, u], [dxf]), 
        "s": cs.Function("fs", [xs, u], [dxs])
    }
    rk4 = {
        "f": cs.Function("RK4f", [xf,h,u],[RK4(f["f"],xf,h,u)]),
        "s": cs.Function("RK4s", [xf,h,u],[RK4(f["s"],xs,h,u)])
    }
    convert = {
        "s": cs.Function("to_flight", [xs],[xfs]),
        "f": cs.Function("to_stance", [xf],[xsf])
    }
    guard = {
        "s" : cs.Function("guards", [xs], [e_LO(xs)]),
        "f" : cs.Function("guardf", [xf], [e_TD(xf)])
    }
    next_mode = {
        "f":"s", 
        "s":"f"
    }
