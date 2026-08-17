import casadi as cs
import numpy as np
from collections import deque
from solver import Continuation_Solver
from matplotlib import pyplot as plt
import generator
import continuation
import dynamics
import plotting

### CONSTANTS

N_F = 25
N_S = 50
K = 40
W = cs.sqrt(10)
N_BRANCH = 10
N_STEPS = 400
N_NEWTON = 10
STEP_SIZE = 0.005
EPSILON = 0.05
PERIODIC = True
U0 = np.array(
    [0.0, 1.01, 0.0, 0.0, 0.0, 0.0, 0.00565685, 0.01525397, 0.00565685, 0.0, 1.01]
)


args_dynamics = (W, K, N_F, N_S)
(
    energy_flight,
    stance_to_flight,
    flight_to_stance,
    traj_f,
    traj_s,
    traj_f_list,
    traj_s_list,
) = dynamics.build_casadi_functions(args_dynamics)
args_continuation = (stance_to_flight, flight_to_stance, traj_f, traj_s, energy_flight)
compute_J, compute_Jz, residual, hessian = continuation.build_casadi_functions(
    args_continuation, PERIODIC
)
continuation_functions = (compute_J, compute_Jz, hessian)

solver = Continuation_Solver(residual)


solver.initialize(U0)
U0 = solver.solve()
to_explore = deque([U0])
explored = []
connected_component = []
idx_branch = 0

while to_explore and idx_branch < N_BRANCH: 
    print(idx_branch)
    u_star = to_explore.popleft()
    M,BP,IP,TP = generator.compute_generator(solver,u_star,STEP_SIZE,N_STEPS,continuation_functions)

    for bifurcation_point in BP:
        new = True
        for special_point in explored:
            if np.linalg.norm(bifurcation_point[0] - special_point) < EPSILON: 
                new = False
        if new: 
            explored.append(bifurcation_point[0])
            u1,u2 = generator.find_regular_point(bifurcation_point)
            to_explore.append(u1)
            to_explore.append(u2)

    for turning_point in TP:
            new = True
            for special_point in explored:
                if np.linalg.norm(turning_point[0] - special_point) < EPSILON: 
                    new = False
            if new: 
                explored.append(turning_point[0])
    

    connected_component.append(M)
    idx_branch += 1

args_plotting = (stance_to_flight,flight_to_stance,traj_f_list,traj_s_list)
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1,projection="3d")
ax.set_proj_type('ortho')
ax.plot([0.,3.],[0,0],[0,0],'k--')
for M in connected_component:
    plotting.plot_generator(ax,M,'red',2,STEP_SIZE,args_plotting,False,5)
# ax.set_xlim(0,3) 
# ax.set_ylim(-0.5,0.5)
# ax.set_zlim(-2,2)
ax.view_init(elev=30, azim=-45, roll=0)
plt.show()
