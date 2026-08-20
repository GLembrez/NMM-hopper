import casadi as cs

class Continuation_Solver():

    def __init__(self,residual):

        self.opti = cs.Opti()
        self.opti.solver(
            "ipopt", {"print_time": 0, "ipopt.print_level": 0, "ipopt.tol": 1e-12}
        )
        self.u = self.opti.variable(11)
        self.opti.subject_to(residual(self.u) == 0)

    def solve(self):

        self.opti.solve()
        return self.opti.value(self.u).copy()

    def initialize(self,u):

        self.opti.set_initial(self.u,u)
