import casadi as cs
import numpy as np
import pickle
from matplotlib import pyplot as plt
from NMM.solver import Continuation_Solver
from MPC.solver import MPCSolver

N_SAMPLES = 100
R = cs.DM([[1,0],[0,1]])
NMM_solver = Continuation_Solver(40,cs.sqrt(10),True)

def get_touch_down(solver,u):
    x0,dt,xi = u[:6],u[6:9],u[9]
    x_TD = solver.traj_f(x0,dt[0],xi)
    return x_TD

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

def build_solver(data,N_SAMPLES,R,reverse=False):
    NMM = data.copy()
    inf = 1
    while inf<NMM.shape[0] and NMM[inf+1,-1]<NMM[inf,-1]:
        inf += 1
    sup = inf 
    while sup<NMM.shape[0] and NMM[sup+1,-1]>NMM[sup,-1]:
        sup += 1
    if reverse:
        NMM[:,[0,2,3,5]] = - NMM[:,[0,2,3,5]]
    NMM_cropped = NMM[inf:sup,:].copy()
    skip = max(1,int(NMM_cropped.shape[0]/N_SAMPLES))
    NMM_sampled = NMM_cropped[::skip,:]
    NMM_touch_down = []
    for i in range(NMM_sampled.shape[0]):
        NMM_touch_down.append(get_touch_down(NMM_solver,NMM_sampled[i,:]))
    NMM_touch_down = np.array(NMM_touch_down).reshape((NMM_sampled.shape[0],-1))
    LUT_x = cs.interpolant('LUTx','bspline',[NMM_sampled[:,-1]],-np.sin(NMM_touch_down[:,2]))
    LUT_y = cs.interpolant('LUTy','bspline',[NMM_sampled[:,-1]],NMM_touch_down[:,1])
    LUT_dx = cs.interpolant('LUTdx','bspline',[NMM_sampled[:,-1]],NMM_touch_down[:,3])
    LUT_dy = cs.interpolant('LUTdy','bspline',[NMM_sampled[:,-1]],NMM_touch_down[:,4])
    LUT_list = [LUT_x,LUT_y,LUT_dx,LUT_dy]
    MPC_solver = MPCSolver(40,cs.sqrt(10),50,R,LUT_list)
    return MPC_solver

data_locomotion = pickle.load(open('data/data_test.pkl', 'rb'))
data_switch = pickle.load(open('data/data_switch.pkl','rb'))
NMM_locomotion = np.array(data_locomotion[4])
NMM_switch = np.array(data_switch[2])

solver_R = build_solver(NMM_locomotion,100,R,False)
solver_L = build_solver(NMM_locomotion,100,R,True)
solver_RL = build_solver(NMM_switch,100,R,False)
solver_LR = build_solver(NMM_switch,100,R,True)

u_star = NMM_locomotion[10]
u_switch = NMM_switch[np.argmin(np.abs(NMM_switch[:,-1]-u_star[-1]))]
traj_star = get_trajectory(NMM_solver,u_star)
traj_switch = get_trajectory(NMM_solver,u_switch)
traj_switch[[0,2,3,5],:] = - traj_switch[[0,2,3,5],:]
interpolator = np.array([i/99 for i in range(100)])
traj_interpolated = traj_star * (1 - interpolator) + traj_switch * interpolator
x0 = traj_star[[0,1,3,4],0]
xs = traj_star[[0,1,3,4],:50]
xf = traj_star[:,50:]
# xf = traj_switch[:,50:].copy()
# xf[[0,2,3,5],:] = - xf[[0,2,3,5],:]
# xf[0,:] += -xf[0,0] + traj_star[0,50]
dt = cs.vertcat(u_star[7],u_switch[6]+u_switch[8])
solver_RL.initialize(x0,xs,xf,dt)
xs_star,xf_star,dt_star,cmd = solver_RL.solve()

# x_TD_star = xf_star[[0,1,3,4],-1]
# x_TD_star[0] = -cs.sin(xf_star[2,-1])
# e = NMM_solver.energy_flight(xf_star[:,-1])
# print(x_TD_star)
# print(cs.vertcat(solver_RL.LUT_x(e),solver_RL.LUT_y(e),solver_RL.LUT_dx(e),solver_RL.LUT_dy(e)))
# print(cs.vertcat(solver_R.LUT_x(e),solver_R.LUT_y(e),solver_R.LUT_dx(e),solver_R.LUT_dy(e)))


fig = plt.figure()
# plt.plot(NMM[:,1],NMM[:,3])
# plt.plot(NMM_sampled[:,1],NMM_sampled[:,3])
# plt.plot(LUT_y(E_linspace),LUT_dx(E_linspace))
plt.plot(xs[0,:],xs[1,:],'blue')
plt.plot(xf[0,:],xf[1,:],'blue')
plt.plot(xs_star[0,:],xs_star[1,:],'red')
plt.plot(xf_star[0,:],xf_star[1,:],'red')
# plt.plot(traj_star[0,:],traj_star[1,:])
plt.plot(traj_switch[0,:],traj_switch[1,:])
# plt.plot(cmd.T)
plt.show()
