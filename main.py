import casadi as cs
import numpy as np
from NMM.solver import Continuation_Solver
from MPC.solver import MPCSolver
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

solver_locomotion = Continuation_Solver(40,1,1000,0.01,True)
solver_switch = Continuation_Solver(40,1,1000,0.01,False)
NMM_locomotion = BFS.search(u0,solver_locomotion,5)
NMM_switch = BFS.search(u0,solver_switch,3)

with open("data/locomotion.pkl", "wb") as file:
    pickle.dump(NMM_locomotion, file)
with open("data/switch.pkl", "wb") as file:
    pickle.dump(NMM_switch, file)

