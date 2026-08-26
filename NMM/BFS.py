import casadi as cs
import numpy as np
import NMM.generator_alt as generator
from collections import deque

def find_regular_point(u,p,jump=0.05):
    return u + jump * p.reshape((11,))

def search(u0,solver,N_BRANCH):
    solver.initialize(u0)
    u = solver.solve()
    to_explore = deque([u])
    explored = []
    connected_component = []
    idx_branch = 0


    while to_explore and idx_branch < N_BRANCH: 
        print(idx_branch)
        u_star = to_explore.popleft()
        M,BP,IP,TP = generator.compute(solver,u_star)

        for bifurcation_point in BP:
            new = True
            for special_point in explored:
                if np.linalg.norm(bifurcation_point[0] - special_point) < solver.EPSILON: 
                    new = False
            if new: 
                explored.append(bifurcation_point[0])
                u_b, p1, p2 = bifurcation_point
                u1,u2 = find_regular_point(u_b,p1), find_regular_point(u_b,p2)
                to_explore.append(u1)
                to_explore.append(u2)

        for turning_point in TP:
                new = True
                for special_point in explored:
                    if np.linalg.norm(turning_point[0] - special_point) < solver.EPSILON: 
                        new = False
                if new: 
                    explored.append(turning_point[0])
                    u_t, p1 = turning_point
                    u1 = find_regular_point(u_t,p1,1)
                    to_explore.append(u1)
        

        connected_component.append(M)
        idx_branch += 1

    return connected_component
