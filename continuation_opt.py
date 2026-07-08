import casadi as cs
import hyperparameters as params
import dynamics


class ContinuationSolver:

    def __init__(self, K, W, periodic=True):

        self.K = K
        self.W = W
        self.opti = cs.Opti()
        self.opti.solver(
            "ipopt", {"print_time": 0, "ipopt.print_level": 0, "ipopt.tol": 1e-12}
        )

        self.energy = self.opti.parameter()

        self.dt_s = self.opti.variable()
        self.dt_f = self.opti.variable()
        self.traj_s = self.opti.variable(4, params.N_S)
        self.traj_f = self.opti.variable(6, params.N_F)

        self.declare_constraints()

        self.set_periodicity(periodic)

    def declare_constraints(self):

        # dynamics constraints
        for i in range(params.N_S - 1):
            self.opti.subject_to(
                self.traj_s[:, i + 1]
                == dynamics.RK4s(self.traj_s[:, i], 0.0, self.dt_s, self.K)
            )
        for i in range(params.N_F - 1):
            self.opti.subject_to(
                self.traj_f[:, i + 1]
                == dynamics.RK4f(self.traj_f[:, i], 0.0, self.dt_f, self.W)
            )

        # touch-down constraint
        self.opti.subject_to(
            self.traj_f[2, -1] == cs.atan(-self.traj_s[0, 0] / self.traj_s[1, 0])
        )
        self.opti.subject_to(self.traj_f[1, -1] == self.traj_s[1, 0])
        self.opti.subject_to(self.traj_f[3:5, -1] == self.traj_s[2:, 0])

        # lift-off constraint
        self.opti.subject_to(self.traj_s[0, -1] ** 2 + self.traj_s[1, -1] ** 2 == 1)

        # energy constraint
        self.opti.subject_to(
            0.5 * (self.traj_f[3, 0] ** 2 + self.traj_f[4, 0] ** 2) + self.traj_f[1, 0]
            == self.energy
        )

    def set_periodicity(self, periodic):
        if periodic:
            self.opti.subject_to(
                self.traj_f[1:, 0] == dynamics.stance_to_flight(self.traj_s[:, -1])[1:]
            )
        else:
            pass

    def initialize(self, trajf, trajs, dtf, dts, Ed):
        self.opti.set_value(self.energy, Ed)
        self.opti.set_initial(self.dt_f, dtf)
        self.opti.set_initial(self.dt_s, dts)
        self.opti.set_initial(self.traj_f, trajf)
        self.opti.set_initial(self.traj_s, trajs)

    def solve(self):
        self.opti.solve()
        return (
            self.opti.value(self.traj_f),
            self.opti.value(self.traj_s),
            self.opti.value(self.dt_f),
            self.opti.value(self.dt_s),
        )
