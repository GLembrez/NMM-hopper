import casadi as cs
import numpy as np
import pickle 
from matplotlib import pyplot as plt
from MPC.solver_1step import Solver 

np.set_printoptions(precision=3,suppress=True)

EMIN,EMAX = 1.5,10.
E0 = 3

with open("data/generator_separation.pkl", "rb") as file:
    generator_separation = pickle.load(file)

points = []

fig = plt.figure()
# ax = fig.add_subplot(1,1,1)

for w_idx in range(5):
    for k_idx in range(5):

        with open("data/dataset_{}{}.pkl".format(k_idx,w_idx), "rb") as file:
            data = pickle.load(file)
        # data = pickle.load(open('data/dataset_4.pkl','rb'))

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
        print("Stiffness: {} | frequency: {}".format(data["K"],data["W"]))
        if sol["success"]:
            print(np.asarray([sol["cost"],sol["wall-time"],sol["iterations"]]))
            for s in generator_separation:
                E,K,W,x_min,f_min = s
                if K==data["K"] and W ==data["W"] and E==E0:
                    points.append((f_min,sol["cost"]))
        else:
            print("Error")

        

plt.plot([p[0] for p in points], [p[1] for p in points], linewidth=0, marker='o')
plt.show()
