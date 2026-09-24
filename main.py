import casadi as cs
import numpy as np
from NMM.solver import Continuation_Solver
import NMM.BFS as BFS
import pickle

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

# u0 = np.array([ 0., 2.438015, -0.000005, -0.727379,  0., 0.686422,  0.068171, 0.010264, 0.068171,  0., 2.702533])
w_list = [3,2.5,2,1.5,1,0.5]
i = 0
for w in w_list:
    solver_locomotion = Continuation_Solver(40,w,1000,0.01,True)
    solver_switch = Continuation_Solver(40,w,1000,0.01,False)
    NMM_locomotion = BFS.search(u0,solver_locomotion,9)
    NMM_switch = BFS.search(u0,solver_switch,5)

    trajectory_space = {
        "K": 40,
        "W": w,
        "STEP_SIZE": 0.01,
        "locomotion": NMM_locomotion,
        "switch": NMM_switch,
    }

    with open("data/dataset_{}.pkl".format(i), "wb") as file:
        pickle.dump(trajectory_space, file)

    i += 1

