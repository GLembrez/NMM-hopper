import casadi as cs
import NMM.dynamics as dynamics
import NMM.continuation as continuation 

class Continuation_Solver:

    def __init__(self, K, W, PERIODIC):

        self.N_F = 25               # number of flight samples
        self.N_S = 50               # number of stance samples
        self.K = K                  # dimensionless stiffness
        self.W = W                  # dimensionless swing frequency
        self.N_STEPS = 50000        # maximum number of steps along a branch
        self.N_NEWTON = 10          # number of newton steps
        self.STEP_SIZE = 0.01       # step size along a branch
        self.EPSILON = 0.05         # minimal distance between special points
        self.PERIODIC = PERIODIC    # toggles antiperiodic steps

        # create solver instance using casadi opti stack
        opts = {"print_time": 0, "ipopt.print_level": 0, "ipopt.tol": 1e-12}
        self.opti = cs.Opti()
        self.opti.solver("ipopt", opts)
        self.u = self.opti.variable(11)

        self.build_dynamics()
        self.build_continuation()
        self.build_constraint()

    def build_dynamics(self):
        dynamics.build(self)

    def build_continuation(self):
        continuation.build(self) 

    def build_constraint(self):
        # add constraint r(u) = 0
        self.opti.subject_to(self.residual(self.u) == 0)


    def solve(self):
        self.opti.solve()
        return self.opti.value(self.u).copy()

    def initialize(self,u):
        self.opti.set_initial(self.u,u)

