import casadi as cs 
import MPC.actuated_dynamics as dynamics 

class Solver:

    def __init__(self,N,K,W,E_MIN,E_MAX,LUT,reverse=False):
        self.LUT = LUT
        self.N = N 
        self.W = W
        self.K = K
        self.E_MIN = E_MIN
        self.E_MAX = E_MAX
        self.reverse = reverse
        self.EVENT_MARGIN = 1e-3
        self.H_MIN = 1e-8
        self.H_MAX = 1e-1

        self.St = cs.diag(cs.DM([1,1,1]))
        self.Ss = cs.diag(cs.DM([1,1,1]))
        self.Sf = cs.diag(cs.DM([1,1,1,1,1]))

        # create solver instance using casadi opti stack
        opts = {"print_time": 0, "ipopt.print_level": 0, "ipopt.tol": 1e-6}
        self.opti = cs.Opti()
        self.opti.solver("ipopt", opts)

        self.x0 = self.opti.parameter(4)
        self.xs = [self.opti.variable(4, N) for _ in range(2)]
        self.xf = [self.opti.variable(6, N) for _ in range(2)]
        self.us = [self.opti.variable(N - 1) for _ in range(2)]
        self.uf = [self.opti.variable(N - 1) for _ in range(2)]
        self.h = self.opti.variable(4)
        self.alpha = self.opti.variable()
        self.hs = self.h[[0, 2]]
        self.hf = self.h[[1, 3]]

        self.opti.subject_to(self.opti.bounded(self.H_MIN, self.h, self.H_MAX))
        self.opti.subject_to(self.opti.bounded(E_MIN, self.alpha, E_MAX))

        self.build_dynamics()
        self.constrain()

    def constrain(self):
        Jm = 0
        Ju = 0

        # look-up target trajectory
        # TODO use surface LUT M(alpha,phi) instead of generating the whole dynamics
        u_star = self.LUT(self.alpha)
        target_f2 = self.traj_f_list(u_star[:6],u_star[6],0)
        target_s = self.traj_s_list(self.flight_to_stance(target_f2[:,-1]),u_star[7],0)
        target_f1 = self.traj_f_list(self.stance_to_flight(target_s[:,-1]),u_star[8],0)
        target_f = cs.horzcat(target_f1,target_f2)
        w = cs.linspace(0, 1, self.N)

        rs = self.Ss @ (self.xs[1][1:,:] - target_s[1:,:])
        rf = self.Sf @ (self.xf[1][1:,:] - target_f[1:,:])
        Jm += self.hs[1] *  cs.sumsqr(cs.diag(w) @ rs.T)
        Jm += self.hf[1] * cs.sumsqr(cs.diag(w) @ rf.T)

        for j in range(2):
            # dynamics constraint
            self.add_dynamics(self.xs[j],self.us[j],self.hs[j],self.RK4s)
            self.add_dynamics(self.xf[j],self.uf[j],self.hf[j],self.RK4f)

            # intra-step continuity
            self.opti.subject_to(self.xf[j][:, 0] == self.stance_to_flight(self.xs[j][:, -1]))

            # lift-off
            self.opti.subject_to(self.liftoff(self.xs[j][:, -1]) == 0)
            self.opti.subject_to(self.liftoff_rate(self.xs[j][:, -1]) >= self.EVENT_MARGIN)

            # touch-down
            self.opti.subject_to(self.touchdown(self.xf[j][:, -1]) == 0)
            self.opti.subject_to(self.touchdown_rate(self.xf[j][:, -1]) <= -self.EVENT_MARGIN)

            # running cost
            Ju += self.hs[j] * cs.sumsqr(self.us[j]) + self.hf[j] * cs.sumsqr(self.uf[j])

        # inter-step continuity
        self.opti.subject_to(self.xs[1][:, 0] == self.flight_to_stance(self.xf[0][:, -1]))

        # initial constraint 
        self.opti.subject_to(self.xs[0][:,0] == self.x0)

        # terminal tube constraint
        xtd = self.flight_to_stance(self.xf[1][:, -1])
        rt = self.St @ (xtd[1:] - target_f[[1,3,4],-1])
        self.opti.subject_to(cs.sumsqr(rt) <= (1e-2)**2)

        if self.reverse:
            xsw = self.xf[0][:, -1]
            self.opti.subject_to( xsw[3] * cs.cos(xsw[2]) + xsw[4] * cs.sin(xsw[2]) == 0)

        self.opti.minimize(Ju + 1e-1 * Jm)

    def build_dynamics(self):
        dynamics.build(self)

    def add_dynamics(self, x, u, h, step):
        for k in range(self.N-1):
            self.opti.subject_to(x[:, k + 1] == step(x[:, k], h, u[k]))

    def initialize(self,x0,xs_guess,xf_guess,h_guess):
        self.opti.set_value(self.x0, x0)
        self.opti.set_initial(self.h, h_guess)
        self.opti.set_initial(self.alpha, 0.5*(x0[2]**2 + x0[3]**2) + x0[1])
        for j in range(2):
            self.opti.set_initial(self.xs[j], xs_guess[j])
            self.opti.set_initial(self.xf[j], xf_guess[j])
            self.opti.set_initial(self.us[j], 0)
            self.opti.set_initial(self.uf[j], 0)

    def solve(self):
        sol = self.opti.solve()
        return {
            "stance": [sol.value(x) for x in self.xs],
            "flight": [sol.value(x) for x in self.xf],
            "stance_control": [sol.value(u) for u in self.us],
            "flight_control": [sol.value(u) for u in self.uf],
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
