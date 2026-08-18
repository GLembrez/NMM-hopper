import casadi as cs
import numpy as np

IDX = [1,5,3]

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

def plot_generator(ax,branch,c,lw,STEP_SIZE,dynamics_function,plot_indiv=False,skip=10,d=1):
    traj = np.array(branch)
    idx = 0
    for i in range(traj.shape[0] -1):
        if np.linalg.norm(traj[i+1,:] - traj[i,:]) > 2*STEP_SIZE:
            idx = i

    if traj[-1,3] > 1e-2 or traj[0,3] > 1e-2:
        ax.plot(traj[:idx+1,IDX[0]],d*traj[:idx+1,IDX[1]],d*traj[:idx+1,IDX[2]],linewidth = lw, color = c)
        ax.plot(traj[idx+1:,IDX[0]],d*traj[idx+1:,IDX[1]],d*traj[idx+1:,IDX[2]],linewidth = lw, color = c)
    elif traj[-1,3] < -1e-2 or traj[0,3] < -1e-2:
        ax.plot(traj[:idx,IDX[0]],-d*traj[:idx,IDX[1]],-d*traj[:idx,IDX[2]],linewidth = lw, color = c)
        ax.plot(traj[idx+1:,IDX[0]],-d*traj[idx+1:,IDX[1]],-d*traj[idx+1:,IDX[2]],linewidth = lw, color = c)

    if plot_indiv:
        for i in range(0,traj.shape[0],skip):
            plot_traj(ax,u2traj(dynamics_function,traj[i,:]),c,0.5)

def prettify3D(ax,xlabel='',ylabel='',zlabel='',title='',fgColor='#cad3f5',bgColor='#24273a',fontsize=14):
    ax.set_xlabel(xlabel,fontsize=fontsize,color=fgColor)
    ax.set_ylabel(ylabel,fontsize=fontsize,color=fgColor)
    ax.set_zlabel(zlabel,fontsize=fontsize,color=fgColor)
    ax.set_title(title,fontsize=fontsize,color=fgColor)
    ax.xaxis.line.set_color(fgColor)
    ax.yaxis.line.set_color(fgColor)
    ax.zaxis.line.set_color(fgColor)
    ax.tick_params(axis='x', colors=fgColor)
    ax.tick_params(axis='y', colors=fgColor)
    ax.tick_params(axis='z', colors=fgColor)
    ax.xaxis.label.set_color(fgColor)
    ax.yaxis.label.set_color(fgColor)
    ax.zaxis.label.set_color(fgColor)
    ax.set_facecolor(bgColor)
    frame = ax.legend(fontsize=fontsize,labelcolor=fgColor).get_frame()
    frame.set_facecolor(bgColor)
    frame.set_edgecolor(fgColor)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(fgColor)
    ax.yaxis.pane.set_edgecolor(fgColor)
    ax.zaxis.pane.set_edgecolor(fgColor)
