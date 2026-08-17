import casadi as cs
import numpy as np

def u2traj(dynamics_function,u):
    stance_to_flight,flight_to_stance,traj_f_list,traj_s_list = dynamics_function
    x0,dt,xi = u[:6],u[6:9],u[9]
    traj1 = traj_f_list(x0, dt[0], xi)
    traj2 = stance_to_flight(
        traj_s_list(flight_to_stance(traj1[:, -1]), dt[1], xi)
    )
    traj3 = traj_f_list(traj2[:, -1], dt[2], xi)
    traj1[0, :] += traj2[0, 0] - traj1[0, -1]
    traj3[0, :] += traj2[0, -1] - traj3[0, 0]
    traj = cs.horzcat(traj1, traj2, traj3)
    return np.array(traj)

def plot_traj(ax,traj,c,lw=0.5):
    ax.plot(traj[1,:],traj[2,:],traj[3,:], linewidth=lw,color=c)

def plot_generator(ax,branch,c,lw,STEP_SIZE,dynamics_function,plot_indiv=False,skip=10):
    traj = np.array(branch)
    idx = 0
    for i in range(traj.shape[0] -1):
        if np.linalg.norm(traj[i+1,:] - traj[i,:]) > 2*STEP_SIZE:
            idx = i
    print(idx)
    if traj[-1,3] > 0.2:
        ax.plot(traj[:idx+1,1],traj[:idx+1,2],traj[:idx+1,3],linewidth = lw, color = c)
        ax.plot(traj[idx+1:,1],traj[idx+1:,2],traj[idx+1:,3],linewidth = lw, color = 'b')
    elif traj[-1,3] < -0.2:
        ax.plot(traj[:idx,1],-traj[:idx,2],-traj[:idx,3],linewidth = lw, color = c)
        ax.plot(traj[idx+1:,1],-traj[idx+1:,2],-traj[idx+1:,3],linewidth = lw, color = 'b')

    if plot_indiv:
        for i in range(0,traj.shape[0],skip):
            plot_traj(ax,u2traj(dynamics_function,traj[i,:]),c,0.5)

