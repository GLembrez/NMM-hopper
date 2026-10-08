import casadi as cs
import numpy as np
import pickle 
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from MPC.solver_1step import Solver 

np.set_printoptions(precision=3,suppress=True)


w_list = np.linspace(1,3,10)
k_list = np.linspace(20,200,10)

EMIN,EMAX = 2,10.
E0 = 6

with open("data/generator_separation.pkl", "rb") as file:
    generator_separation = pickle.load(file)

points = []
grid_bool = np.zeros((10,10))
grid_sep  = np.zeros((10,10))



for w_idx in range(10):
    for k_idx in range(10):

        with open("data/dataset_{}{}.pkl".format(k_idx,w_idx), "rb") as file:
            data = pickle.load(file)
        # data = pickle.load(open('data/dataset_4.pkl','rb'))

        FW = np.array(data["locomotion"][2])
        SW = np.array(data["switch"][2])

        if FW[100,3] < 0 :
            FW[:,[0,2,3,5]] = - FW[:,[0,2,3,5]]
        if SW[100,3] < 0 :
            SW[:,[0,2,3,5]] = - SW[:,[0,2,3,5]]

        E_min = max(EMIN, max(min(FW[:,-1]),min(SW[:,-1])))
        E_max =  min(EMAX, min(max(FW[:,-1]),max(SW[:,-1])))
        E_grid = np.linspace(E_min, E_max, 100)
        fw_sampled = []
        sw_sampled = []
        for i in range(100):
            E = E_grid[i]
            idx_fw = np.argwhere(FW[:,-1]>E)
            idx_sw = np.argwhere(SW[:,-1]>E)
            idx_fw = -1 if len(idx_fw)==0 else idx_fw[0][0]
            idx_sw = -1 if len(idx_sw)==0 else idx_sw[0][0]
            fw_sampled.append(FW[idx_fw,:])
            sw_sampled.append(SW[idx_sw,:])
        fw_sampled = np.array(fw_sampled).ravel(order="C")
        sw_sampled = np.array(sw_sampled).ravel(order="C")
        LUT_FW = cs.interpolant('LUT_FW', 'bspline',[E_grid],fw_sampled)
        LUT_SW = cs.interpolant('LUT_SW', 'bspline',[E_grid],sw_sampled)

        solver = Solver(50,data["K"],data["W"],max(E0-2,E_min), min(E0+2,E_max),LUT_SW,True)

        u_star = LUT_FW(E0)
        traj1 = solver.traj_f_list(u_star[:6],u_star[6],0)
        traj2 = solver.traj_s_list(solver.flight_to_stance(traj1[:,-1]),u_star[7],0)
        traj3 = solver.traj_f_list(solver.stance_to_flight(traj2[:,-1]),u_star[6],0)
        traj4 = solver.traj_f_list(traj3[:,-1],u_star[6],0)

        # ax.plot(traj1[0,:].T,traj1[1,:].T)
        # ax.plot(traj2[0,:].T,traj2[1,:].T)
        # ax.plot(traj3[0,:].T,traj3[1,:].T)
        # ax.plot(traj4[0,:].T,traj4[1,:].T)

        solver.initialize(
            traj2[:,0],
            traj2,
            np.hstack([traj3,traj4]),
            cs.DM([u_star[7],u_star[6]+u_star[8]])
        )
        sol = solver.solve()
        if sol["success"] and sol["cost"] < 10:
            # print(np.asarray([sol["cost"],sol["wall-time"],sol["iterations"]]))
            for s in generator_separation:
                E,K,W,x_min,f_min = s
                if K==data["K"] and W ==data["W"] and E==E0:
                    points.append((f_min,sol["cost"]))
        else:
            print("Error at {:2f},{:2f}".format(data["K"],data["W"]))
            grid_bool[k_idx,w_idx] = 1

        for s in generator_separation:
            E,K,W,x_min,f_min = s
            if K==data["K"] and W ==data["W"] and E==E0:
                grid_sep[k_idx,w_idx] = f_min


import matplotlib.ticker as ticker

print(grid_sep)
mask_cmap = ListedColormap([
    (0, 0, 0, 0),      # False: transparent
    (1, 0, 0, 0.45),   # True: transparent red
])
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1)
im = ax.imshow(grid_sep)
fig.colorbar(im,ax=ax)
# ax.imshow(
#     grid_bool.astype(int),
#     cmap=mask_cmap,
#     vmin=0,
#     vmax=1,
# )
ax.set_xticks(range(10),labels=[f"{k:.2f}" for k in k_list])
ax.set_yticks(range(10),labels=[f"{w:.2f}" for w in w_list])
plt.savefig('heatmap.svg')


print(len(points))

plt.figure()
plt.plot([p[0] for p in points], [p[1] for p in points], linewidth=0, marker='o',color='#e09d2e')
plt.ylim(-1,10)
plt.grid()
plt.show()
