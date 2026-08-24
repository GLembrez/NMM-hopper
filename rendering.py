import pickle 
from matplotlib import pyplot as plt
import casadi as cs
import numpy as np
import NMM.generator as generator
import NMM.continuation as continuation
import NMM.dynamics as dynamics
from NMM.solver import Continuation_Solver
import plotting


NMM_solver = Continuation_Solver(40,cs.sqrt(10),True)

data_locomotion = pickle.load(open('data/data_test.pkl', 'rb'))
data_switch = pickle.load(open('data/data_switch.pkl', 'rb'))

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1,projection="3d")
ax.set_proj_type('ortho')
for M in data_switch:
    plotting.plot_generator(ax,M,'red',2,0.01,NMM_solver,True,40,1)
for M in data_locomotion:
    plotting.plot_generator(ax,M,'blue',2,0.01,NMM_solver,True,400,1)
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
