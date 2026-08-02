import casadi as cs
import hyperparameters as params
import dynamics as dn
import numpy as np

N_F = params.N_F
N_S = params.N_S


class ContinuationSolver:

    def __init__(self, periodic):

        self.periodic = periodic
        self.opti = cs.Opti()
        self.opti.solver(
            "ipopt", {"print_time": 0, "ipopt.print_level": 0, "ipopt.tol": 1e-12}
        )

        self.dt = self.opti.variable(3)
        self.traj_apex_TD = self.opti.variable(6, N_F)
        self.traj_TD_LO = self.opti.variable(4, N_S)
        self.traj_LO_apex = self.opti.variable(6, N_F)
        self.xi = self.opti.variable(2)

        self.Ed = self.opti.parameter()

        self.declare_constraints()
        self.set_periodicity()

    def declare_constraints(self):
        for i in range(N_F - 1):
            # flight dn constraint
            self.opti.subject_to(
                self.traj_apex_TD[:, i + 1]
                == dn.RK4f(self.traj_apex_TD[:, i], self.dt[0], self.xi[0])
            )
            self.opti.subject_to(
                self.traj_LO_apex[:, i + 1]
                == dn.RK4f(self.traj_LO_apex[:, i], self.dt[2], self.xi[0])
            )

        for i in range(N_S - 1):
            # stance dn constraint
            self.opti.subject_to(
                self.traj_TD_LO[:, i + 1]
                == dn.RK4s(self.traj_TD_LO[:, i], self.dt[1], self.xi[1])
            )

        # touch-down
        self.opti.subject_to(
            self.traj_apex_TD[[1,2,3,4], - 1] == dn.stance_to_flight(self.traj_TD_LO[:, 0])[[1,2,3,4],:]
        )  # continuity of angular velocity ?
        self.opti.subject_to(
            self.traj_apex_TD[1, -1] == cs.cos(self.traj_apex_TD[2, -1])
        )

        # lift-off
        self.opti.subject_to(
            dn.stance_to_flight(self.traj_TD_LO[:, - 1])[[1,2,3,4,5], 0] == self.traj_LO_apex[[1,2,3,4,5], 0]
        )
        self.opti.subject_to(
            self.traj_TD_LO[0, - 1] ** 2 + self.traj_TD_LO[1, - 1] ** 2 == 1
        )

        # apex
        self.opti.subject_to(self.traj_apex_TD[4, 0] == 0)

        self.opti.subject_to(dn.energy_flight(self.traj_apex_TD[:, 0]) == self.Ed)

    def set_periodicity(self):
        if self.periodic:
            # periodicity
            self.opti.subject_to(self.traj_apex_TD[1:, 0] == self.traj_LO_apex[1:, -1])
        else:
            self.opti.subject_to(self.traj_apex_TD[[1, 4], 0] == self.traj_LO_apex[[1, 4], -1])
            self.opti.subject_to(self.traj_apex_TD[[2, 3, 5], 0] == -self.traj_LO_apex[[2, 3, 5], -1])

    def initialize(self, traj, dt, Ed):
        self.opti.set_initial(self.traj_apex_TD, traj[:,:N_F])
        self.opti.set_initial(self.traj_TD_LO, dn.flight_to_stance(traj[:,N_F:N_F+N_S]))
        self.opti.set_initial(self.traj_LO_apex, traj[:,N_F+N_S:2*N_F+N_S])
        self.opti.set_initial(self.dt, dt)
        self.opti.set_initial(self.xi, [0.0, 0.0])
        self.opti.set_value(self.Ed, Ed)

    def solve(self):
        self.opti.solve()
        traj = cs.horzcat(
            self.opti.value(self.traj_apex_TD), 
            dn.stance_to_flight(self.opti.value(self.traj_TD_LO)), 
            self.opti.value(self.traj_LO_apex)
        )
        traj[0, :N_F] += traj[0, N_F] - traj[0, N_F - 1]
        traj[0, N_F + N_S :] += -traj[0, N_F + N_S] + traj[0, N_F + N_S - 1]
        return (
            np.array(traj),
            self.opti.value(self.dt),
            self.opti.value(self.xi[0]),
        )
