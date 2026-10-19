from scipy.optimize import brentq

class Real_Time:

    def __init__(self,dynamics,x0,m0):
        self.dynamics = dynamics
        self.state = x0
        self.mode = m0 

    def measure(self):
        return self.state,self.mode

    def apply_control(self,u,Ts):
        x = self.state
        m = self.mode
        x_next = self.dynamics.rk4[m](x,Ts,u)
        g0 = self.dynamics.guard[m](x)
        g1 = self.dynamics.guard[m](x_next)
        if (g0 < 0 and g1 >= 0):
            def residual(tau):
                x_tau = self.dynamics.rk4[m](x,tau,u)
                return self.dynamics.guard[m](x_tau)
            tau_event = brentq(residual,0.0,Ts,xtol=1e-6,rtol=1e-6)
            x = self.dynamics.convert[m](self.dynamics.rk4[m](x,tau_event,u))
            m = self.dynamics.next_mode[m]
            x_next = self.dynamics.rk4[m](x,Ts-tau_event,0.0)

        self.state = x_next
        self.mode = m 



