import pickle 
from matplotlib import pyplot as plt
import casadi as cs
import numpy as np
from NMM.solver import Continuation_Solver
import plotting


NMM_solver = Continuation_Solver(40,1,1000,0.01,True)

data_locomotion = pickle.load(open('data/locomotion.pkl', 'rb'))
data_switch = pickle.load(open('data/switch.pkl', 'rb'))

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1,projection="3d")
ax.set_proj_type('ortho')
surface_switch = np.zeros([100,3,100])
for i in range(100):
    surface_switch[i,:,:] = plotting.u2traj(NMM_solver,data_switch[2][10*i])[[1,2,3],:]
# for M in data_switch:
#     plotting.plot_generator(ax,M,'red',2,0.01,NMM_solver,True,50,1)
surface_locomotion = np.zeros([100,3,100])
for i in range(100):
    surface_locomotion[i,:,:] = plotting.u2traj(NMM_solver,data_locomotion[2][10*i])[[1,2,3],:]

# for M in [data_locomotion[2]]:
#     plotting.plot_generator(ax,M,'blue',2,0.01,NMM_solver,True,50,1)
    # plotting.plot_generator(ax,M,'blue',2,STEP_SIZE,args_plotting,False,1,-1)
# NMM = data_locomotion[2]
# det_list = []
# for u in NMM:
#     z = u[:10]
#     E = u[10]
#     det_list.append(np.linalg.det(NMM_solver.compute_Jz(z,E)))
x_min,x_max = ax.get_xlim()
ax.plot([0.,x_max],[0,0],[0,0],'k--')
ax.plot_surface(surface_locomotion[:,0], surface_locomotion[:,1], surface_locomotion[:,2], cmap='viridis', edgecolor='none', alpha=0.5)
ax.plot_surface(surface_switch[:,0], surface_switch[:,1], surface_switch[:,2], cmap='plasma', edgecolor='none', alpha=0.5)
# ax.set_xlim(0,2) 
# ax.set_ylim(-0.5,0.5)
# ax.set_zlim(-2,2)
ax.view_init(elev=15, azim=-130, roll=0)
ax.set_box_aspect((4, 4,4))
plotting.prettify3D(ax,fgColor='black',bgColor='white')
# plt.plot(det_list)
plt.show()
