import numpy as np

def energy_sample(M,grid):
    idx = np.argsort(M[:,-1])
    M = M[idx,:]
    E = M[:,-1]
    return np.column_stack([np.interp(grid, E, M[:, c]) for c in range(M.shape[1])])
        
def trajectory_sample(u,grid,dynamics):
    x0 = u[:6]
    th,xh = dynamics.integrate_until_event(x0,"f")
    tTD,xTD = th[0,-1],dynamics.convert["f"](xh[:,-1])
    th_s,xh_s = dynamics.integrate_until_event(xTD,"s")
    tLO,xLO = th_s[0,-1],dynamics.convert["s"](xh_s[:,-1])
    th_f,xh_f = dynamics.integrate_until_event(xLO,"f")
    ts_norm = th_s / th_s[0,-1]
    tf_norm = th_f / th_f[0,-1] 

    traj_s = np.column_stack([
        np.interp(grid, ts_norm.squeeze(), xh_s[r,:]) for r in range(xh_s.shape[0])
    ]+[np.interp(grid, ts_norm.squeeze(), th_s.squeeze())])
    traj_f = np.column_stack([
        np.interp(grid, tf_norm.squeeze(), xh_f[r,:]) for r in range(xh_f.shape[0])
    ]+[np.interp(grid, tf_norm.squeeze(), th_f.squeeze())])
    return traj_s,traj_f

def surface(M,grid,dynamics):
    N = M.shape[0]
    surf_s,surf_f = np.zeros((N,N,5)),np.zeros((N,N,7))
    for i in range(N):
        ts,tf = trajectory_sample(M[i],grid,dynamics)
        surf_s[:,i,:] = ts
        surf_f[:,i,:] = tf
    return surf_s,surf_f
