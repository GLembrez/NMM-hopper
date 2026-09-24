import casadi as cs 
import MPC.actuated_dynamics as dynamics 

class Solver:

    def __init__(self,N,K,W,E_MIN,E_MAX,LUT,reverse=False):
        self.LUT = LUT
        self.N = N 
        self.W = W
        self.K = K
        self.E_MIN = 2.8
        self.E_MAX = 3.2
        self.reverse = reverse
        self.EVENT_MARGIN = 1e-3
        self.H_MIN = 1e-8
        self.H_MAX = 1e-1

        self.St = cs.diag(cs.DM([1,0.01,0.01]))

        # create solver instance using casadi opti stack
        opts = {"print_time": 0, "ipopt.print_level": 0, "ipopt.tol": 1e-6}
        self.opti = cs.Opti()
        self.opti.solver("ipopt", opts)

        self.x0 = self.opti.parameter(4)
        self.xs = self.opti.variable(4, N)
        self.xf = self.opti.variable(6, N)
        self.us = self.opti.variable(N - 1)
        self.uf = self.opti.variable(N - 1)
        self.h = self.opti.variable(2)
        self.alpha = self.opti.variable()
        self.hs = self.h[0]
        self.hf = self.h[1]

        self.opti.subject_to(self.opti.bounded(self.H_MIN, self.h, self.H_MAX))
        self.opti.subject_to(self.opti.bounded(E_MIN, self.alpha, E_MAX))

        self.build_dynamics()
        self.constrain()

    def constrain(self):
        Ju = 0

        # look-up target trajectory
        u_star = self.LUT(self.alpha)
        target_f = self.traj_f_list(u_star[:6],u_star[6],0)

        # dynamics constraint
        self.add_dynamics(self.xs,self.us,self.hs,self.RK4s)
        self.add_dynamics(self.xf,self.uf,self.hf,self.RK4f)

        # intra-step continuity
        self.opti.subject_to(self.xf[:, 0] == self.stance_to_flight(self.xs[:, -1]))

        # lift-off
        self.opti.subject_to(self.liftoff(self.xs[:, -1]) == 0)
        self.opti.subject_to(self.liftoff_rate(self.xs[:, -1]) >= self.EVENT_MARGIN)

        # touch-down
        self.opti.subject_to(self.touchdown(self.xf[:, -1]) == 0)
        self.opti.subject_to(self.touchdown_rate(self.xf[:, -1]) <= -self.EVENT_MARGIN)

        # running cost
        Ju += self.hs * cs.sumsqr(self.us) + self.hf * cs.sumsqr(self.uf)

        # initial constraint 
        self.opti.subject_to(self.xs[:,0] == self.x0[:])

        # terminal tube constraint
        xtd = self.flight_to_stance(self.xf[:, -1])
        self.rt =  (xtd[1:] - target_f[[1,3,4],-1])
        # self.rt = target_f[:,-1]
        self.opti.subject_to(cs.sumsqr(self.rt) <= (1e-2)**2)

        self.opti.minimize(Ju)

    def build_dynamics(self):
        dynamics.build(self)

    def add_dynamics(self, x, u, h, step):
        for k in range(self.N-1):
            self.opti.subject_to(x[:, k + 1] == step(x[:, k], h, u[k]))

    def initialize(self,x0,xs_guess,xf_guess,h_guess,us_guess=0,uf_guess=0):
        self.opti.set_value(self.x0, x0)
        self.opti.set_initial(self.h, h_guess)
        self.opti.set_initial(self.alpha, 0.5*(x0[2]**2 + x0[3]**2) + x0[1])
        self.opti.set_initial(self.xs, xs_guess)
        self.opti.set_initial(self.xf, xf_guess)
        self.opti.set_initial(self.us, us_guess)
        self.opti.set_initial(self.uf, uf_guess)

    def solve(self):
        sol = self.opti.solve()
        return {
            "stance": sol.value(self.xs),
            "flight": sol.value(self.xf),
            "stance_control": sol.value(self.us),
            "flight_control": sol.value(self.uf),
            "step_size": sol.value(self.h),
            "alpha": sol.value(self.alpha),
        }
    
    @staticmethod
    def liftoff(x):
        return x[0]**2 + x[1]**2 - 1

    @staticmethod
    def touchdown(x):
        return x[1] - cs.cos(x[2])

    @staticmethod
    def liftoff_rate(x):
        return x[0] * x[2] + x[1] * x[3]

    @staticmethod
    def touchdown_rate(x):
        return x[4] + cs.sin(x[2]) * x[5]
