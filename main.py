import casadi as cs
import numpy as np
from NMM.solver import Continuation_Solver
from MPC.solver import MPCSolver
import NMM.BFS as BFS

u0 = np.array([
    0.0,        # px
    1.01,       # py
    0.0,        # theta
    0.0,        # dpx
    0.0,        # dpy
    0.0,        # dtheta
    0.00566,    # flight duration
    0.0152,     # stance duration 
    0.00566,     # flight duration
    0.0,        # xi
    1.01        # energy
])

solver = Continuation_Solver(40,cs.sqrt(10),False)
NMM_locomotion = BFS.search(u0,solver,3)
