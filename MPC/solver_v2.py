import casadi as cs
import MPC.actuated_dynamics as dynamics


class MPCSolver:

    def __init__(self, K, W, N, switch, LUT,Emin, Emax):

        self.N = N  # number of samples in each phase
        self.K = K  # dimensionless stiffness
        self.W = W  # dimensionless swing frequency
        self.switch = switch
        self.Emin = Emin
        self.Emax = Emax

        # create solver instance using casadi opti stack
        opts = {"print_time": 0, "ipopt.print_level": 0, "ipopt.tol": 1e-6}
        self.opti = cs.Opti()
        self.opti.solver("ipopt", opts, {"max_iter": 500})

        self.x0 = self.opti.parameter(4)
        self.xs = self.opti.variable(4, 2*N)
        self.xf = self.opti.variable(6, 2*N)
        self.dt = self.opti.variable(4)
        self.us = self.opti.variable(2*N)
        self.uf = self.opti.variable(2*N)
        self.alpha = self.opti.variable()
        self.LUT = LUT

        self.build_dynamics()
        self.constraints()

    def build_dynamics(self):
        dynamics.build(self)

    def constraints(self):
        N = self.N 
        self.J = 0

        # look up table
        # self.opti.subject_to(self.Emin < self.alpha)
        # self.opti.subject_to(self.Emax > self.alpha)
        # u_star = self.LUT(self.alpha)
        # xf0_ref = self.traj_f_list(u_star[:6],u_star[6],0)
        # xs_ref = self.traj_s_list(self.flight_to_stance(xf0_ref[:,-1]),u_star[7],0)
        # xf1_ref = self.traj_f_list(self.stance_to_flight(xs_ref[:,-1]),u_star[8],0)
        # xf2_ref = self.traj_f_list(xf1_ref[:,-1],u_star[6],0)

        for i in range(N - 1):
            # dynamics constraint
            self.opti.subject_to(
                self.xs[:, i + 1] == self.RK4s(self.xs[:, i], self.dt[0], self.us[i])
            )
            self.opti.subject_to(
                self.xf[:, i + 1] == self.RK4f(self.xf[:, i], self.dt[1], self.uf[i])
            )
            self.opti.subject_to(
                self.xs[:, N + i + 1] == self.RK4s(self.xs[:,N + i], self.dt[2], self.us[N+i])
            )
            self.opti.subject_to(
                self.xf[:,N+ i + 1] == self.RK4f(self.xf[:, N+i], self.dt[3], self.uf[N+i])
            )
            self.J += self.us[i] ** 2 + self.uf[i]**2
            self.J += self.us[N+i] ** 2 + self.uf[N+i]**2

        # initial condition
        self.opti.subject_to(self.xs[:, 0] == self.x0)

        # lift-off
        self.opti.subject_to(self.xs[0, N-1] ** 2 + self.xs[1, N-1] ** 2 == 1)
        self.opti.subject_to(self.xs[0, -1] ** 2 + self.xs[1, -1] ** 2 == 1)

        self.opti.subject_to(self.xf[[0,1,3,4], 0] == self.xs[:, N-1])
        self.opti.subject_to(self.xf[2,0] == cs.atan(-self.xs[0,N-1]/self.xs[1,N-1]))
        self.opti.subject_to(self.xf[5,0] == self.xs[0,N-1]*self.xs[3,N-1] - self.xs[1,N-1]*self.xs[2,N-1])

        self.opti.subject_to(self.xf[[0,1,3,4], N] == self.xs[:, -1])
        self.opti.subject_to(self.xf[2,N] == cs.atan(-self.xs[0,-1]/self.xs[1,-1]))
        self.opti.subject_to(self.xf[5,N] == self.xs[0,-1]*self.xs[3,-1] - self.xs[1,-1]*self.xs[2,-1])

        # terminal constraint
        self.opti.subject_to(cs.cos(self.xf[2, -1]) == self.xf[1, -1])
        self.opti.subject_to(cs.cos(self.xf[2, N-1]) == self.xf[1, N-1])

        self.opti.subject_to(self.xs[[1,2,3],self.N] == self.xf[[1,3,4],self.N-1])
        self.opti.subject_to(self.xs[0,self.N] == -cs.sin(self.xf[2,self.N-1]))
        
        # running cost
        # self.opti.minimize(self.J)


        # if self.switch:
        #     self.opti.subject_to(self.xf[3,N-1]*cs.cos(self.xf[2,N-1]) + self.xf[4,N-1]*cs.sin(self.xf[2,N-1]) == 0)

    def initialize(self, x0, xs, xf, dt):
        self.opti.set_value(self.x0, x0)
        self.opti.set_initial(self.xs, xs)
        self.opti.set_initial(self.xf, xf)
        self.opti.set_initial(self.dt, dt)
        self.opti.set_initial(self.us, cs.GenDM_zeros(2*self.N))
        self.opti.set_initial(self.uf, cs.GenDM_zeros(2*self.N))
        self.opti.set_initial(self.alpha,0.5*(xf[3,0]**2 + xf[4,0]**2) + xf[1,0])

    def solve(self):
        self.opti.solve()
        return (
            self.opti.value(self.xs),
            self.opti.value(self.xf),
            self.opti.value(self.dt),
            self.opti.value(self.us),
            self.opti.value(self.uf),
        )
