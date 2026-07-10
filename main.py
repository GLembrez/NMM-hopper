import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import casadi as cs
    from scipy.linalg import eig
    from matplotlib import pyplot as plt
    from time import time
    import sys

    import simple_shooting as ss
    import dynamics
    import hyperparameters as params
    import continuation_functions as utils
    from continuation_opt import ContinuationSolver


    return ContinuationSolver, cs, dynamics, np, params, plt, ss, time, utils


@app.cell
def _(np):
    np.set_printoptions(precision=3)
    return


@app.cell
def _(ContinuationSolver, cs):
    K = 40
    W = cs.sqrt(10)

    solver = ContinuationSolver(K,W)
    return K, W, solver


@app.cell
def _(K, W, np, utils):
    x0 = np.array([0.,1.,0.,0.,0.1,0.])
    trajf_initial,trajs_initial,dtf_initial,dts_initial = utils.init_trajectory(x0,K,W)
    return dtf_initial, dts_initial, trajf_initial, trajs_initial


@app.cell
def _(dynamics, utils):
    def record(branch,trajf,trajs,dtf,dts):
        branch["traj_f"].append(trajf)
        branch["traj_s"].append(dynamics.stance_to_flight(trajs))
        branch["dt_f"].append(dtf)
        branch["dt_s"].append(dts)
        branch["E"].append(utils.energy_flight(trajf[:,0]))

    return (record,)


@app.cell
def _(
    K,
    W,
    cs,
    dtf_initial,
    dts_initial,
    np,
    params,
    plt,
    record,
    solver,
    time,
    trajf_initial,
    trajs_initial,
    utils,
):
    branch = {"E": [], "dir": [], "traj_f": [], "traj_s": [], "dt_f": [], "dt_s": []}


    _E = utils.energy_flight(trajf_initial[:, 0])
    solver.initialize(trajf_initial, trajs_initial, dtf_initial, dts_initial, _E)
    _xfh, _xsh, _dtf, _dts = solver.solve()
    N = 2
    time_eval=0
    mul_list = []
    for i in range(N):
        apex = _xfh[:, params.N_F // 2]
        _var = cs.vertcat(_xfh[:, -1], _xsh[:, -1], apex, _dtf, _dts, _dtf)
        _xfh, _xsh, _dtf, _dts, mul, _, _dir = utils.initialize_next(_var, apex, np.array([1,0,0,0]),0.025, K, W)
        mul_list.append(mul)
        _E = utils.energy_flight(_xfh[:, 0])
        # if np.abs(mul)<1e-1:
        #     print(mul,_dir,_var,_E)
        start = time()
        solver.initialize(_xfh, _xsh, _dtf, _dts,_E)
        _xfh, _xsh, _dtf, _dts = solver.solve()
        end = time()
        time_eval += (end-start)/N
        record(branch,_xfh, _xsh, _dtf, _dts)
    plt.plot(mul_list)
    return N, apex, branch


@app.cell
def _(K, W, apex, np, plt, solver, ss, utils):
    _var = np.array( [0, 0.999963, 0, 0, -0.714195, 0, 0, 1, 0, 0.714143, 0, 1.255, 0, 0, -2.5977e-05, 0, 0.0285667, 0.0115425, 0.0285667])
    _dir = np.array([ 0.,    -0.021, -0.64,  -0.768])
    _E = utils.energy_flight(_var[:6])
    _xfh, _xsh, _dtf, _dts = ss.trajectory(_var,K,W)
    solver.initialize(_xfh, _xsh, _dtf, _dts,_E)
    _xfh, _xsh, _dtf, _dts = solver.solve()
    dist = 0.05
    if utils.energy_apex(apex[[1, 2, 3, 5]] + dist * _dir) < utils.energy_apex(apex[[1, 2, 3, 5]]):
            dist = -dist
    _apex = _var[10:16]
    _apex[[1,2,3,5]] += dist * _dir 
    _var = ss.full_newton(_var,_apex,K,W)
    _xfh, _xsh, _dtf, _dts = ss.trajectory(_var,K,W)


    plt.plot(_xfh[4,:].T)
    plt.plot(_xfh[5,:].T)

    # solver.initialize(_xfh, _xsh, _dtf, _dts,_E)
    # _xfh, _xsh, _dtf, _dts = solver.solve()



    # plt.plot(_xfh[0,:].T,_xfh[1,:].T)
    # plt.plot(_xsh[0,:].T,_xsh[1,:].T)


    # _N = 0
    # for _i in range(_N):
    #     _apex = _xfh[:, params.N_F // 2]
    #     _var = cs.vertcat(_xfh[:, -1], _xsh[:, -1], _apex, _dtf, _dts, _dtf)
    #     print(_E)
    #     _xfh, _xsh, _dtf, _dts, _mul, _dir,_ = utils.initialize_next(_var, apex, _dir,dist, K, W)
    #     _E = utils.energy_flight(_xfh[:, 0])
    #     solver.initialize(_xfh, _xsh, _dtf, _dts,_E)
    #     _xfh, _xsh, _dtf, _dts = solver.solve()
    return


@app.cell
def _(N, branch, plt):
    for idx in range(N):
        plt.plot(branch["traj_f"][idx][1,:].T,branch["traj_f"][idx][4,:].T)
        plt.plot(branch["traj_s"][idx][1,:].T,branch["traj_s"][idx][4,:].T)
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
