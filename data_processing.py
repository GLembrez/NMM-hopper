import casadi as cs
import numpy as np
from scipy.optimize import minimize_scalar
from NMM.solver import Continuation_Solver
from matplotlib import pyplot as plt
import pickle

np.set_printoptions(precision=3)

EMIN,EMAX = 2,10.

# fig = plt.figure(figsize=(10,10))
# ax = fig.add_subplot(1,1,1,projection='3d')


LUT_list = []
for k_idx in range(10):
    LUT_list.append([])
    for w_idx in range(10):

        with open("data/dataset_{}{}.pkl".format(k_idx,w_idx), "rb") as file:
            data = pickle.load(file)

        solver = Continuation_Solver(data["K"],data["W"],1000,0.01,True)
        FW = np.array(data["locomotion"][2])
        SW = np.array(data["switch"][2])
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
            TD_FW = solver.traj_f(FW[idx_fw,:6],FW[idx_fw,6],0)
            TD_SW = solver.traj_f(SW[idx_sw,:6],SW[idx_sw,6],0)
            fw_sampled.append(TD_FW)
            sw_sampled.append(TD_SW)
        fw_sampled = np.array(fw_sampled).ravel(order="C")
        sw_sampled = np.array(sw_sampled).ravel(order="C")
        LUT_FW = cs.interpolant('LUT_FW', 'bspline',[E_grid],fw_sampled)
        LUT_SW = cs.interpolant('LUT_SW', 'bspline',[E_grid],sw_sampled)
        LUT_list[k_idx].append((solver,LUT_FW,LUT_SW,E_min,E_max))

        to_plot = []
        to_plot_2 = []
        for e in E_grid[:-2]:
            to_plot.append((LUT_FW(e)[4],LUT_FW(e)[2],LUT_FW(e)[3]))
            to_plot_2.append((LUT_SW(e)[4],LUT_SW(e)[2],LUT_SW(e)[3]))
#         ax.plot([t[0] for t in to_plot],[t[1] for t in to_plot],[t[2] for t in to_plot])
#         ax.plot([t[0] for t in to_plot_2],[t[1] for t in to_plot_2],[t[2] for t in to_plot_2],'--')

# plt.show()


w_list = np.linspace(1,3,10)
k_list = np.linspace(15,60,10)


D = np.diag([0,1,0,1,1,0])
generator_separation = np.zeros((10,10))
E0 = 4.5
for k_idx in range(10):
    for w_idx in range(10):
        solver,LUT_FW,LUT_SW,E_min,E_max = LUT_list[k_idx][w_idx]
        gamma1 = LUT_FW(E0)
        to_minimize = lambda e : np.linalg.norm(D @ (gamma1 - LUT_SW(e)))
        minimizer = minimize_scalar(
            to_minimize,
            bounds=(max(E0-2,E_min), min(E0+2,E_max)),
            method="bounded",
        )
        x_min = minimizer.x
        f_min = minimizer.fun
        generator_separation[k_idx,w_idx] = f_min

        # generator_separation.append((E0,solver.K,solver.W,x_min,f_min))
        # print(E_min, x_min, f_min, solver.K, solver.W)

print(generator_separation)

fig = plt.figure()
plt.imshow(generator_separation)
plt.xticks(range(10),labels=k_list)
plt.yticks(range(10),labels=w_list)
plt.colorbar()
plt.show()

# with open("data/generator_separation.pkl", "wb") as file:
#     pickle.dump(generator_separation, file)

# with open("data/dataset_95.pkl", "rb") as file:
#             data = pickle.load(file)

# solver = Continuation_Solver(data["K"],data["W"],1000,0.01,True)
# FW = np.array(data["locomotion"][6])
# SW = np.array(data["switch"][4])
# E_min = max(EMIN, max(min(FW[:,-1]),min(SW[:,-1])))
# E_max =  min(EMAX, min(max(FW[:,-1]),max(SW[:,-1])))
# E_grid = np.linspace(E_min, E_max, 100)
# fw_sampled = []
# sw_sampled = []
# for i in range(100):
#     E = E_grid[i]
#     idx_fw = np.argwhere(FW[:,-1]>E)
#     idx_sw = np.argwhere(SW[:,-1]>E)
#     idx_fw = -1 if len(idx_fw)==0 else idx_fw[0][0]
#     idx_sw = -1 if len(idx_sw)==0 else idx_sw[0][0]
#     TD_FW = solver.traj_f(FW[idx_fw,:6],FW[idx_fw,6],0)
#     TD_SW = solver.traj_f(SW[idx_sw,:6],SW[idx_sw,6],0)
#     fw_sampled.append(TD_FW)
#     sw_sampled.append(TD_SW)
# fw_sampled = np.array(fw_sampled).ravel(order="C")
# sw_sampled = np.array(sw_sampled).ravel(order="C")
# LUT_FW = cs.interpolant('LUT_FW', 'bspline',[E_grid],fw_sampled)
# LUT_SW = cs.interpolant('LUT_SW', 'bspline',[E_grid],sw_sampled) 

# E0 = 4.5
# D = np.diag([0,1,0,1,1,0])
# gamma1 = LUT_FW(E0)
# to_minimize = lambda e : np.linalg.norm(D @ (gamma1 - LUT_SW(e)))
# minimizer = minimize_scalar(
#     to_minimize,
#     bounds=(max(E0-2,E_min), min(E0+2,E_max)),
#     method="bounded",
# )
# x_min = minimizer.x
# f_min = minimizer.fun
# print(f_min)
