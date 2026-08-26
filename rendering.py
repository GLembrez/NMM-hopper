import pickle 
from matplotlib import pyplot as plt
import casadi as cs
import numpy as np
from NMM.solver import Continuation_Solver
import plotting

np.set_printoptions(precision=6, suppress=True)

NMM_solver = Continuation_Solver(40,cs.sqrt(10),1000,0.01,True)

data_locomotion = pickle.load(open('data/data_locomotion.pkl', 'rb'))
data_switch = pickle.load(open('data/data_switch.pkl', 'rb'))
data_test = pickle.load(open('data/data_test.pkl', 'rb'))

print(np.array(data_locomotion[8][0]))

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1,projection="3d")
ax.set_proj_type('ortho')
# surface_switch = np.zeros([100,3,100])
# for i in range(100):
#     surface_switch[i,:,:] = plotting.u2traj(NMM_solver,data_switch[2][10*i])[[1,2,3],:]
for M in data_switch:
    plotting.plot_generator(ax,M,'red',2,0.01,NMM_solver,False,50,1)

# for M in data_test:
#     plotting.plot_generator(ax,M,'green',2,0.01,NMM_solver,False,100,1)
# for M in [data_test[1]]:
#     plotting.plot_generator(ax,M,'red',2,0.01,NMM_solver,False,100,1)
# surface_locomotion = np.zeros([100,3,100])
# for i in range(100):
#     surface_locomotion[i,:,:] = plotting.u2traj(NMM_solver,data_locomotion[2][10*i])[[1,2,3],:]

for M in data_locomotion:
    plotting.plot_generator(ax,M,'blue',2,0.01,NMM_solver,False,50,1)
    # plotting.plot_generator(ax,M,'blue',2,STEP_SIZE,args_plotting,False,1,-1)
# NMM = data_locomotion[2]
# det_list = []
# for u in NMM:
#     z = u[:10]
#     E = u[10]
#     det_list.append(np.linalg.det(NMM_solver.compute_Jz(z,E)))
# det_list = np.array(det_list)
# x_min,x_max = ax.get_xlim()
# ax.plot([0.,x_max],[0,0],[0,0],'k--')
# ax.plot_surface(surface_locomotion[:,0], surface_locomotion[:,1], surface_locomotion[:,2], cmap='viridis', edgecolor='none', alpha=0.5)
# ax.plot_surface(surface_switch[:,0], surface_switch[:,1], surface_switch[:,2], cmap='plasma', edgecolor='none', alpha=0.5)
# ax.set_xlim(0,2) 
# ax.set_ylim(-0.5,0.5)
# ax.set_zlim(-2,2)
# ax.view_init(elev=15, azim=-130, roll=0)
# ax.set_box_aspect((4, 4,4))
# plotting.prettify3D(ax,fgColor='black',bgColor='white')
# plt.plot(np.sign(det_list[:-1] * det_list[1:])[:-100] * 1e8)
# plt.plot(det_list[:-100])
plt.show()
