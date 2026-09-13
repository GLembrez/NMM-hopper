import casadi as cs
import numpy as np
import pickle
from matplotlib import pyplot as plt
from NMM.solver import Continuation_Solver
from MPC.solver import MPCSolver

N_SAMPLES = 100
R = cs.DM([[1,0],[0,1]])
NMM_solver = Continuation_Solver(40,1,True)

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
    while inf<NMM.shape[0]-1 and NMM[inf+1,-1]<NMM[inf,-1]:
        inf += 1
    sup = inf 
    while sup<NMM.shape[0]-1 and NMM[sup+1,-1]>NMM[sup,-1]:
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
    MPC_solver = MPCSolver(40,1,50,R,LUT_list)
    return MPC_solver

data_locomotion = pickle.load(open('data/locomotion4.pkl', 'rb'))
data_switch = pickle.load(open('data/switch4.pkl','rb'))
NMM_locomotion = np.array(data_locomotion[2])
NMM_switch = np.array(data_switch[2])

solver_R = build_solver(NMM_locomotion,100,R,False)
solver_L = build_solver(NMM_locomotion,100,R,True)
solver_RL = build_solver(NMM_switch,100,R,False)
solver_LR = build_solver(NMM_switch,100,R,True)

u_star = NMM_locomotion[100]
u_switch = NMM_switch[np.argmin(np.abs(NMM_switch[:,-1]-u_star[-1]))]
traj_star = get_trajectory(NMM_solver,u_star)
traj_switch = get_trajectory(NMM_solver,u_switch)


x01 = traj_star[[0,1,3,4],0]
xs1 = traj_star[[0,1,3,4],:50]
xf1 = traj_star[:,50:]
dt1 = cs.vertcat(u_star[7],u_star[6]+u_star[8])
solver_RL.initialize(x01,xs1,xf1,dt1)
xs_star,xf_star,dt_star,cmd = solver_RL.solve()

x02 = xf_star[[0,1,3,4],-1]
x02[0] = -cs.sin(xf_star[2,-1])
xs2 = traj_switch[[0,1,3,4],:50]
xf2 = traj_star[:,50:]
xf2[[0,2,3,5],:] = -xf2[[0,2,3,5],:] 
xf2[0,:] += xs2[0,-1] - xf2[0,0]
dt2 = cs.vertcat(u_switch[7],u_switch[6]+u_switch[8])
# x0[0] += -cs.sin(xf_star[2,-1])-x0[0]
# xs[0,:] += -cs.sin(xf_star[2,-1])-xs[0,0]
# xf[0,:] += -cs.sin(xf_star[2,-1])-xf[0,0]


solver_L.initialize(x02,xs2,xf2,dt2)
e = NMM_solver.energy_flight(xf_star[:,-1])
print(solver_L.LUT_x(e),solver_L.LUT_y(e),solver_L.LUT_dx(e),solver_L.LUT_dy(e))
print(xf2[:,-1])
print(x02)
print(xs2[:,0])
xs_star1,xf_star1,dt_star1,cmd1 = solver_L.solve()

traj_next = NMM_solver.traj_s_list(NMM_solver.flight_to_stance(xf_star1[:,-1]),dt_star1[0],0)
traj_next[0,:] += xf_star[0,-1] - traj_next[0,0]

fig = plt.figure()
# plt.plot(NMM[:,1],NMM[:,3])
# plt.plot(NMM_sampled[:,1],NMM_sampled[:,3])
# plt.plot(LUT_y(E_linspace),LUT_dx(E_linspace))
# plt.plot(xs1[0,:],xs1[1,:],'teal')
# plt.plot(xf1[0,:],xf1[1,:],'teal')
# plt.plot(xs2[0,:],xs2[1,:],'blue')
# plt.plot(xf2[0,:],xf2[1,:],'blue')
plt.plot(xs_star[0,:],xs_star[1,:],'red')
plt.plot(xf_star[0,:],xf_star[1,:],'red')
plt.plot(xs_star1[0,:],xs_star1[1,:],'green')
plt.plot(xf_star1[0,:],xf_star1[1,:],'green')
# plt.plot(traj_star[0,:],traj_star[1,:])
# plt.plot(traj_switch[0,:],traj_switch[1,:])
plt.plot(traj_next[0,:].T,traj_next[1,:].T)
# plt.plot(cmd.T)
plt.show()
