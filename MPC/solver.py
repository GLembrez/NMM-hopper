import casadi as cs
import MPC.actuated_dynamics as dynamics


class MPCSolver:

    def __init__(self, K, W, N, R, LUT_list):

        self.N = N  # number of samples in each phase
        self.K = K  # dimensionless stiffness
        self.W = W  # dimensionless swing frequency
        self.R = R  # QR weight (Q = 0)

        # create solver instance using casadi opti stack
        opts = {"print_time": 0, "ipopt.print_level": 0, "ipopt.tol": 1e-12}
        self.opti = cs.Opti()
        self.opti.solver("ipopt", opts)

        self.x0 = self.opti.parameter(4)
        self.xs = self.opti.variable(4, self.N)
        self.xf = self.opti.variable(6, self.N)
        self.dt = self.opti.variable(2)
        self.u = self.opti.variable(2, N)
        self.alpha = self.opti.variable()
        self.LUT_x = LUT_list[0]
        self.LUT_y = LUT_list[1]
        self.LUT_dx = LUT_list[2]
        self.LUT_dy = LUT_list[3]

        self.build_dynamics()
        self.constraints()

    def build_dynamics(self):
        dynamics.build(self)

    def constraints(self):
        self.J = 0
        for i in range(self.N - 1):
            # dynamics constraint
            self.opti.subject_to(
                self.xs[:, i + 1] == self.RK4s(self.xs[:, i], self.dt[0], self.u[0, i])
            )
            self.opti.subject_to(
                self.xf[:, i + 1] == self.RK4f(self.xf[:, i], self.dt[1], self.u[1, i])
            )
            self.J += self.u[:, i].T @ self.R @ self.u[:, i]

        # initial condition
        self.opti.subject_to(self.xs[:, 0] == self.x0)

        # lift-off
        self.opti.subject_to(self.xs[0, -1] ** 2 + self.xs[1, -1] ** 2 == 1)
        self.opti.subject_to(self.xf[[0,1,3,4], 0] == self.xs[:, -1])
        self.opti.subject_to(self.xf[2,0] == cs.atan(-self.xs[0,-1]/self.xs[1,-1]))
        self.opti.subject_to(self.xf[5,0] == self.xs[0,-1]*self.xs[3,-1] - self.xs[1,-1]*self.xs[2,-1])

        # terminal constraint
        self.opti.subject_to(cs.cos(self.xf[2, -1]) == self.xf[1, -1])
        x_TD = cs.vertcat(
            -cs.sin(self.xf[2, -1]), cs.cos(self.xf[2, -1]), self.xf[3, -1], self.xf[4, -1]
        )
        # E_TD = 0.5 * (x_TD[2] ** 2 + x_TD[3] ** 2) + x_TD[1]
        gamma = cs.vertcat(
            self.LUT_x(self.alpha), 
            self.LUT_y(self.alpha), 
            self.LUT_dx(self.alpha), 
            self.LUT_dy(self.alpha)
        )
        # self.opti.subject_to(cs.sqrt((x_TD - gamma).T @ (x_TD - gamma) )<= 1e-1)
        self.opti.subject_to(x_TD == gamma)

        # # running cost
        self.opti.minimize(self.J)

    def initialize(self, x0, xs, xf, dt):
        self.opti.set_value(self.x0, x0)
        self.opti.set_initial(self.xs, xs)
        self.opti.set_initial(self.xf, xf)
        self.opti.set_initial(self.dt, dt)
        self.opti.set_initial(self.u, cs.GenDM_zeros(2, self.N))
        self.opti.set_initial(self.alpha,0.5*(xf[3,0]**2 + xf[4,0]**2) + xf[1,0])

    def solve(self):
        self.opti.solve()
        return (
            self.opti.value(self.xs),
            self.opti.value(self.xf),
            self.opti.value(self.dt),
            self.opti.value(self.u),
        )
