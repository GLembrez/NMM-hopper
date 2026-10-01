import casadi as cs
import numpy as np
import pickle
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from MPC.baseline import Solver as Baseline
from MPC.solver_1step import Solver as Solver_1

def reorder(branch):
    traj = np.array(branch)
    idx = 0
    for i in range(traj.shape[0] -1):
        if np.linalg.norm(traj[i+1,:] - traj[i,:]) > 0.02:
            idx = i+1
    fixed_traj = np.vstack([np.flip(traj[:idx,:],axis=0),traj[idx:,:]])
    if traj[-1,3] < -1e-2 or traj[0,3] < -1e-2:
        fixed_traj[:,[0,2,3,5]] = - fixed_traj[:,[0,2,3,5]]
    return fixed_traj

def separate(traj):
    idx = 0
    for i in range(100,traj.shape[0] -101):
        de1 = traj[i,-1] - traj[i-1,-1]
        de2 = traj[i+1,-1] - traj[i,-1]
        if de1*de2 < 0:
            idx = i + 1
    return traj[:idx,:],traj[idx:,:]

def flip(traj):
    flipped_traj = np.zeros(traj.shape)
    for i in range(traj.shape[0]):
        flipped_traj[i] = traj[traj.shape[0]-1-i]
    return flipped_traj


def getLUT(NMM):
    N_SAMPLES = 100
    skip = max(1,int(NMM.shape[0]/N_SAMPLES))
    NMM_sampled = NMM[10:-10:skip,:]
    E_grid = NMM_sampled[:,-1]
    values = NMM_sampled.ravel(order="C")
    LUT = cs.interpolant('LUT', 'bspline',[E_grid],values)
    return LUT,E_grid[0],E_grid[-1]

def get_trajectory(solver,u):
    x0,dt,xi = u[:6],u[6:9],u[9]
    traj1 = solver.traj_f_list(x0, dt[0], xi)
    traj2 = solver.stance_to_flight(
        solver.traj_s_list(solver.flight_to_stance(traj1[:, -1]), dt[1], xi)
    )
    traj3 = solver.traj_f_list(traj2[:, -1], dt[2], xi)
    traj4 = solver.traj_f_list(traj3[:,-1], dt[0], xi)
    traj = cs.horzcat(traj2, traj2[:,-1], traj3[:,:-1], traj4)
    return np.array(traj)

data = pickle.load(open('data/dataset_4.pkl','rb'))
K = data["K"]
W = data["W"]

FW = reorder(data["locomotion"][2])
FW_TO_BW = np.flip(reorder(data["switch"][2]),axis = 0)
FW1,FW2 = separate(FW)
FW = FW1 if FW1.shape[0] > FW2.shape[0] else FW2
BW = FW.copy()
BW[:,[0,2,3,5]] = - FW[:,[0,2,3,5]]
BW_TO_FW = FW_TO_BW.copy()
BW_TO_FW[:,[0,2,3,5]] = - FW_TO_BW[:,[0,2,3,5]]

LUT_FW,Emin_FW,Emax_FW = getLUT(FW)
LUT_BW,Emin_BW,Emax_BW = getLUT(BW)
LUT_FW_TO_BW,Emin_FW_TO_BW,Emax_FW_TO_BW = getLUT(FW_TO_BW)
LUT_BW_TO_FW,Emin_BW_TO_FW,Emax_BW_TO_FW = getLUT(BW_TO_FW)

N=20
E_list = np.linspace(3,6,N)
J1,J2,Jb = [],[],[]
T1,T2,Tb = [],[],[]
PS1,PF1,PS2,PF2,PSb,PFb = [],[],[],[],[],[]
for E_star in E_list:

    Emin, Emax = E_star - 2, E_star + 2
    S_rev_1s = Solver_1(50,K,W,max(Emin,Emin_BW_TO_FW),min(Emax_BW_TO_FW,Emax),LUT_FW_TO_BW,True)
    S_bac_1s = Solver_1(50,K,W,max(Emin_BW,Emin),min(Emax_BW,Emax),LUT_BW,False)
    solver_B = Baseline(50,K,W,max(Emin_BW,Emin),min(Emax_BW,Emax),LUT_BW,False)

    u_fw = LUT_FW(E_star)
    u_bw = LUT_BW(E_star)
    u_fw_to_bw = LUT_FW_TO_BW(E_star)
    u_bw_to_fw = LUT_BW_TO_FW(E_star)
    fw = get_trajectory(S_bac_1s,u_fw)
    bw = get_trajectory(S_bac_1s,u_bw)
    fw_to_bw = get_trajectory(S_bac_1s,u_fw_to_bw)
    bw_to_fw = get_trajectory(S_bac_1s,u_bw_to_fw)

    x0 = fw[[0,1,3,4],0]
    xs_init = fw[[0,1,3,4],:S_rev_1s.N]
    xf_init = fw[:,S_rev_1s.N:]
    dt_init = cs.vertcat(u_fw[7],u_fw[6]+u_fw[8])
    S_rev_1s.initialize(x0,xs_init,xf_init,dt_init,0,0)
    sol_rev_1s = S_rev_1s.solve()
    J1.append(sol_rev_1s["cost"])
    T1.append(sol_rev_1s["wall-time"])
    PS1.append(max(np.abs(sol_rev_1s["stance_control"])))
    PF1.append(max(np.abs(sol_rev_1s["flight_control"])))

    x0 = S_rev_1s.flight_to_stance(sol_rev_1s["flight"][:,-1])
    u_fw_to_bw_fitted = LUT_FW_TO_BW(sol_rev_1s["alpha"])
    fw_to_bw_fitted = get_trajectory(S_bac_1s,u_fw_to_bw_fitted)
    xs_init = fw_to_bw_fitted[[0,1,3,4],:S_rev_1s.N]
    xf_init = fw_to_bw_fitted[:,S_rev_1s.N:]
    dt_init = cs.vertcat(u_fw_to_bw_fitted[7],u_fw_to_bw_fitted[6]+u_fw_to_bw_fitted[8])
    S_bac_1s.initialize(x0,xs_init,xf_init,dt_init)
    sol_bac_1s = S_bac_1s.solve()
    J2.append(sol_bac_1s["cost"])
    T2.append(sol_bac_1s["wall-time"])
    PS2.append(max(np.abs(sol_bac_1s["stance_control"])))
    PF2.append(max(np.abs(sol_bac_1s["flight_control"])))

    x0 = fw[[0,1,3,4],0]
    xs_init = [fw[[0,1,3,4],:S_rev_1s.N],fw[[0,1,3,4],:S_rev_1s.N]]
    xf_init = [fw[:,S_rev_1s.N:],fw[:,S_rev_1s.N:]]
    dt_init = cs.vertcat(u_fw[7],u_fw[6]+u_fw[8],u_fw[7],u_fw[6]+u_fw[8])
    solver_B.initialize(x0,xs_init,xf_init,dt_init)
    sol_baseline = solver_B.solve()
    Jb.append(sol_baseline["cost"])
    Tb.append(sol_baseline["wall-time"])
    PSb.append(max(max(np.abs(sol_baseline["stance_control"][0])),max(np.abs(sol_baseline["stance_control"][1]))))
    PFb.append(max(max(np.abs(sol_baseline["flight_control"][0])),max(np.abs(sol_baseline["flight_control"][1]))))

plt.figure()
plt.plot(Tb,'o',linewidth=0)
plt.plot([T1[i]+T2[i] for i in range(N)],'o',linewidth=0)
plt.show()
