import pickle 
from matplotlib import pyplot as plt
import casadi as cs
import numpy as np
import generator
import continuation
import dynamics
import plotting


### CONSTANTS

N_F = 25
N_S = 50
K = 40
W = cs.sqrt(10)
N_BRANCH = 5
N_STEPS = 20000
N_NEWTON = 10
STEP_SIZE = 0.01
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
compute_J, compute_Jz, residual, hessian,newton = continuation.build_casadi_functions(
    args_continuation, PERIODIC
)
continuation_functions = (compute_J, compute_Jz, hessian)

data_locomotion = pickle.load(open('data_locomotion.pkl', 'rb'))
data_switch = pickle.load(open('data_switch.pkl', 'rb'))

args_plotting = (stance_to_flight,flight_to_stance,traj_f_list,traj_s_list)
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1,projection="3d")
ax.set_proj_type('ortho')
for M in data_switch:
    plotting.plot_generator(ax,M,'red',2,STEP_SIZE,args_plotting,False,40,1)
for M in data_locomotion:
    plotting.plot_generator(ax,M,'blue',2,STEP_SIZE,args_plotting,False,400,1)
    # plotting.plot_generator(ax,M,'blue',2,STEP_SIZE,args_plotting,False,1,-1)
x_min,x_max = ax.get_xlim()
# ax.plot([0.,x_max],[0,0],[0,0],'k--')
# ax.set_xlim(0,3) 
# ax.set_ylim(-0.5,0.5)
# ax.set_zlim(-2,2)
ax.view_init(elev=15, azim=-130, roll=0)
ax.set_box_aspect((4, 4,4))
plotting.prettify3D(ax,fgColor='black',bgColor='white')
plt.show()
