import casadi as cs
import numpy as np
import pickle
from matplotlib import pyplot as plt
from solver import Continuation_Solver

solver = Continuation_Solver(40,cs.sqrt(10),True)
N_SAMPLES = 100
R = cs.DM([[1,0],[0,1]])

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
    traj1[0, :] += traj3[0, -1] - traj1[0, 0]
    traj = cs.horzcat(traj2, traj3, traj1)
    return np.array(traj)

data_locomotion = pickle.load(open('data/data_test.pkl', 'rb'))
NMM = np.array(data_locomotion[2])

N = 20000
skip = max(1,int(N/N_SAMPLES))
NMM_sampled = NMM[:N:skip,:]
NMM_touch_down = []
for i in range(N_SAMPLES):
    NMM_touch_down.append(get_touch_down(solver,NMM_sampled[i,:]))
NMM_touch_down = np.array(NMM_touch_down).reshape((N_SAMPLES,-1))

u_star = NMM_sampled[10,:]
x_TD = get_touch_down(solver,u_star)
traj_star = get_trajectory(solver,u_star)


LUT_x = cs.interpolant('LUTx','bspline',[NMM_sampled[:,-1]],NMM_touch_down[:,0])
LUT_y = cs.interpolant('LUTy','bspline',[NMM_sampled[:,-1]],NMM_touch_down[:,1])
LUT_dx = cs.interpolant('LUTdx','bspline',[NMM_sampled[:,-1]],NMM_touch_down[:,3])
LUT_dy = cs.interpolant('LUTdy','bspline',[NMM_sampled[:,-1]],NMM_touch_down[:,4])
E_linspace = np.linspace(np.min(NMM_sampled[:,-1]),np.max(NMM_sampled[:,-1]),1000)


fig = plt.figure()
plt.plot(NMM[:,1],NMM[:,3])
plt.plot(NMM_sampled[:,1],NMM_sampled[:,3])
plt.plot(LUT_y(E_linspace),LUT_dx(E_linspace))
plt.show()
