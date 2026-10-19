import OCP_layout as Layout
import problem_definition 
import casadi as cs

class Solver:
    def __init__(self,target,sequence,N,dim,bounds):
        self.name = f"{target.name()}_{''.join(sequence)}"
        self.target = target
        self.sequence = sequence
        self.layout = Layout(list(sequence),N,dim)
        problem = problem_definition.make_problem(self.layout,target)
        self.linearize,self.values, self.qp_structure, self.con_min, self.con_max, self.ng = problem
        self.qp_solver = cs.conic(
            f"qp_{self.name}",
            "hpipm",
            self.qp_structure,
            {
                "N":                self.layout.N_ocp, 
                "nx":               self.layout.nx, 
                "nu":               self.layout.nu, 
                "ng":               self.ng,
                "print_time":       False, 
                "verbose":          False, 
                "error_on_fail":    False,
                "hpipm": {
                    "mode":         "balance",
                    "warm_start":   1,
                },
            },
        )
        self.w_min,self.w_max = problem_definition.make_bounds(self.layout,*bounds)
        self.delta =            problem_definition.make_delta(self.layout)
