import casadi as cs
import numpy as np
from NMM.solver import Continuation_Solver
import NMM.BFS as BFS
import pickle
from scipy.integrate import solve_ivp

x0 = np.array([0.0,1.01,0.0,0.0,0.0,0.0])

w_list = np.linspace(1,3,10)
k_list = np.linspace(15,60,10)

def integrate(x0,fun,event):
    return solve_ivp(
        fun = fun,
        t_span = (0.,10.),
        max_step = 1e-2,
        y0 = x0,
        method='DOP853',
        rtol=1e-12,
        atol=1e-12,
        events=(event),
        vectorized=True
    )

def initialize(x0,solver):
    ff = lambda t,x : np.array(solver.ff(x,0)).squeeze()
    fs = lambda t,x : np.array(solver.fs(x,0)).squeeze()
    eTD = lambda t,x : np.cos(x[2]) - x[1]
    eLO = lambda t,x : np.sqrt(x[0]**2 + x[1]**2) - 1
    eTD.terminal = True  
    eTD.direction = 1
    eLO.terminal = True
    eLO.direction = 1
    solTD = integrate(x0,ff,eTD)
    solLO = integrate(
        np.array(solver.flight_to_stance(solTD.y_events[0][0])).squeeze(),fs,eLO
    )
    return np.concatenate([x0,np.array([
        solTD.t_events[0][0] / solver.N_F,
        solLO.t_events[0][0] / solver.N_S,
        solTD.t_events[0][0] / solver.N_F,
        0.,
        x0[1]
    ])])



i=0
for w in w_list:
    j=0
    for k in k_list:

        solver_locomotion = Continuation_Solver(k,w,1000,0.01,True)
        solver_switch = Continuation_Solver(k,w,1000,0.01,False)

        u0 = initialize(x0,solver_locomotion)
        solver_locomotion.initialize(u0)
        u_star = solver_locomotion.solve()

        NMM_locomotion = BFS.search(u_star,solver_locomotion,7)
        NMM_switch = BFS.search(u_star,solver_switch,5)
    

        trajectory_space = {
            "K": k,
            "W": w,
            "Nf": solver_locomotion.N_F,
            "Ns": solver_locomotion.N_S,
            "STEP_SIZE": 0.01,
            "locomotion": NMM_locomotion,
            "switch": NMM_switch,
        }

        with open("data/dataset_{}{}.pkl".format(i,j), "wb") as file:
            pickle.dump(trajectory_space, file)

        j += 1
    i += 1

