import casadi as cs
import hyperparameters as params
import dynamics

N_F = params.N_F
N_S = params.N_S

class ContinuationSolver():

    def __init__(self,periodic):

        self.periodic = periodic
        self.opti = cs.Opti()
        self.opti.solver(
            "ipopt", {"print_time": 0, "ipopt.print_level": 0, "ipopt.tol": 1e-12}
        )

        self.dt = self.opti.variable(2)
        self.traj = self.opti.variable(6, 2 * N_F + N_S)

        self.Ed = self.opti.parameter()

        self.declare_constraints()
        self.set_periodicity()

    def declare_constraints(self):
        for i in range(N_F - 1):
            # flight dynamics constraint
            self.opti.subject_to(
                self.traj[:, i + 1] == dynamics.RK4f(self.traj[:, i], self.dt[0])
            )
            self.opti.subject_to(
                self.traj[:, N_F + N_S + i + 1]
                == dynamics.RK4f(self.traj[:, N_F + N_S + i], self.dt[0])
            )

        for i in range(N_S - 1):
            # stance dynamics constraint
            self.opti.subject_to(
                self.traj[:, N_F + i + 1]
                == dynamics.stance_to_flight(dynamics.RK4s(self.traj[[0, 1, 3, 4], N_F + i], self.dt[1]))
            )

        # touch-down
        self.opti.subject_to(self.traj[:, N_F - 1] == self.traj[:,N_F]) # continuity of angular velocity ?
        self.opti.subject_to(self.traj[1,N_F-1] == cs.cos(self.traj[2,N_F-1]))

        # lift-off
        self.opti.subject_to(self.traj[:, N_F + N_S - 1] == self.traj[:, N_F + N_S])
        self.opti.subject_to(self.traj[0,N_F+N_S-1]**2 + self.traj[1,N_F+N_S-1]**2  == 1)

        # apex 
        self.opti.subject_to(self.traj[4,0]==0)

        self.opti.subject_to(dynamics.energy_flight(self.traj[:, 0]) == self.Ed)

    def set_periodicity(self):
        if self.periodic:
            # periodicity
            self.opti.subject_to(self.traj[[1,2,3,5], 0] == self.traj[[1,2,3,5], -1])
        else:
            self.opti.subject_to(self.traj[1,0] == self.traj[1,-1])
            self.opti.subject_to(self.traj[[2,3,5],0] == -self.traj[[2,3,5],-1])

    def initialize(self, traj, dt, Ed):
        self.opti.set_initial(self.traj, traj)
        self.opti.set_initial(self.dt, dt)
        self.opti.set_value(self.Ed, Ed)

    def solve(self):
        self.opti.solve()
        return (
            self.opti.value(self.traj),
            self.opti.value(self.dt),
            self.opti.value(self.Ed),
        )
