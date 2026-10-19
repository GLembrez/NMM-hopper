import numpy as np
import casadi as cs
from scipy.optimize import brentq

class Hopper:

    def __init__(self, K, W, Nf, Ns):
        self.K = K
        self.W = W
        self.Nf = Nf
        self.Ns = Ns
        self.build_dynamics()

    def build_dynamics(self):
        xs = cs.SX.sym("xs", 4)
        xf = cs.SX.sym("xf", 6)
        u = cs.SX.sym("u")
        h = cs.SX.sym("h")

        l = cs.sqrt(xs[0] ** 2 + xs[1] ** 2)
        dxs = cs.vertcat(
            xs[2:], 
            (self.K * (1 - l) + u) * xs[0] / l, 
            (self.K * (1 - l) + u) * xs[1] / l - 1
        )
        dxf = cs.vertcat(
            xf[3:], 
            0, 
            -1, 
            -(self.W**2) * xf[2] + u
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
        de_TD = -cs.sin(xf[2])*xf[5] - xf[4]
        de_LO = xs[0]*xs[2] + xs[1]*xs[3] 

        self.f =  {
            "f": cs.Function("ff", [xf, u], [dxf]), 
            "s": cs.Function("fs", [xs, u], [dxs])
        }
        self.rk4 = {
            "f": cs.Function("RK4f", [xf,h,u],[self.RK4(self.f["f"],xf,h,u)]),
            "s": cs.Function("RK4s", [xs,h,u],[self.RK4(self.f["s"],xs,h,u)])
        }
        self.convert = {
            "s": cs.Function("to_flight", [xs],[xfs]),
            "f": cs.Function("to_stance", [xf],[xsf])
        }
        self.guard = {
            "s" : cs.Function("guards", [xs], [e_LO]),
            "f" : cs.Function("guardf", [xf], [e_TD])
        }
        self.guard_rate = {
            "s" : cs.Function("guards", [xs], [de_LO]),
            "f" : cs.Function("guardf", [xf], [de_TD])
        }
        self.next_mode = {
            "f":"s", 
            "s":"f"
        }
        self.dim = {
            "f":6, 
            "s":4
        }
        self.N = {
            "f":self.Nf, 
            "s":self.Ns
        }

    def integrate_until_event(self,x0,mode,N_max=1000,h=1e-2):
        x = x0
        t = 0.0
        xh = []
        th = []
        armed = False
        
        for i in range(N_max):

            xh.append(x)
            th.append(t)
            x_next = self.rk4[mode](x,h,0.0)

            g0 = self.guard[mode](x)
            g1 = self.guard[mode](x_next)

            if (not armed) and g1 < 0 and g0<0:
                armed = True

            if g0 * g1 < 0 and armed:
                def residual(tau):
                    x_tau = self.rk4[mode](x,tau,0.0)
                    return self.guard[mode](x_tau)
                
                tau_event = brentq(residual,0.0,h,xtol=1e-6,rtol=1e-6)
                x_event = self.rk4[mode](x,tau_event,0.0)
                xh.append(x_event)
                th.append(t+tau_event)

                return np.column_stack(th),np.column_stack(xh)

            t += h
            x = x_next 

    @staticmethod
    def RK4(f,x,h,u):
        k1 = f(x,u)
        k2 = f(x+0.5*h*k1,u)
        k3 = f(x+0.5*h*k2,u)
        k4 = f(x+h*k3,u)
        x_next = x + h/6 * (k1 + 2*k2 + 2*k3 +k4)
        return x_next
