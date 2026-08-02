import marimo

__generated_with = "0.23.14"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import casadi as cs
    from matplotlib import pyplot as plt
    from time import time
    from collections import deque
    import sys

    import simple_shooting as ss
    import dynamics
    import hyperparameters as params
    import continuation_functions as utils
    from solver import ContinuationSolver

    return ContinuationSolver, cs, dynamics, np, plt, ss, utils


@app.cell
def _(np):
    np.set_printoptions(precision=6,suppress=True)
    return


@app.cell
def _(ContinuationSolver, np):
    periodic = True
    n_steps = 400
    n_branch = 10
    dE_min = 1e-6
    dE_max = 1e-2
    floquet_threshold = 0.15
    x0 = np.array([0.0, 2.2, 0.0, 0.0, 0.0, 0.0])
    solver = ContinuationSolver(periodic)
    return dE_max, dE_min, floquet_threshold, n_steps, periodic, solver, x0


@app.cell
def _(cs, np, periodic, ss):
    def floquet_analysis(traj, dt):
        z = ss.full_newton(cs.vertcat(traj[1:, 0], dt), traj[1:, 0])
        if periodic:
            M = ss.monodromy(traj[1:, -1], z[:5], z[5:])
        else:
            M = ss.monodromy_reverse(traj[1:, -1], z[:5], z[5:])
        val,vec = np.linalg.eig(M)
        idx_sorted = np.argsort(np.abs(val))
        return val[idx_sorted], vec[:,idx_sorted]

    return (floquet_analysis,)


@app.cell
def _(dynamics, floquet_analysis, np, solver, utils):
    def simple_continuation(traj,dt,v,n_steps,dE):
        branch = {"traj": [], "E": [], "dt": []}
        floquet_multipliers = []
        E = float(dynamics.energy_flight(traj[:, 0]))
        utils.register(traj, dt, E, branch)
        x0 = traj[:, 0].copy()
        x0[[1, 2, 3, 5]] += 1e-3 * np.real(v)
        traj, dt = utils.init_trajectory(x0)
        solver.initialize(traj, dt, branch["E"][-1] + dE)
        traj, dt, E = solver.solve()
        utils.register(traj, dt, E, branch)


        for step in range(n_steps):
            traj, dt = utils.estimation(
                branch["traj"][-1], branch["dt"][-1], branch["traj"][-2], branch["dt"][-2]
            )
            solver.initialize(traj, dt, branch["E"][-1] + dE)
            try:
                traj, dt, E = solver.solve()
            except:
                print("solver failed | end of branch")
                break
            utils.register(traj, dt, E, branch)
            val, vec = floquet_analysis(traj, dt)
            floquet_multipliers.append(val)
        
        return branch, floquet_multipliers

    return (simple_continuation,)


@app.cell
def _(
    dE_max,
    dE_min,
    dynamics,
    floquet_analysis,
    floquet_threshold,
    np,
    solver,
    utils,
):
    def explore_component(traj, dt, v, n_steps):

        branch = {"traj": [], "E": [], "dt": []}
        to_explore = []
        floquet_multipliers = []
        previous_val = 10.0
        dE = dE_min
        bifurcation = traj[1:, 0]
        E = float(dynamics.energy_flight(traj[:, 0]))
        utils.register(traj, dt, E, branch)
        x0 = traj[:, 0].copy()
        x0[[1, 2, 3, 5]] += 1e-3 * np.real(v)
        traj, dt = utils.init_trajectory(x0)
        solver.initialize(traj, dt, branch["E"][-1] + dE)
        traj, dt, E = solver.solve()
        utils.register(traj, dt, E, branch)

        for step in range(n_steps):

            if (traj[1, :] < 0).any():
                print("collision with ground | end of branch")
                break

            traj, dt = utils.estimation(
                branch["traj"][-1], branch["dt"][-1], branch["traj"][-2], branch["dt"][-2]
            )
            solver.initialize(traj, dt, branch["E"][-1] + dE)
            try:
                traj, dt, E = solver.solve()
            except:
                print("solver failed | end of branch")
                break
            utils.register(traj, dt, E, branch)

            val, vec = floquet_analysis(traj, dt)
            floquet_multipliers.append(val)
            distance_bifurcation = np.linalg.norm(bifurcation - traj[1:, 0])
            if (
                np.abs(val[1]) < floquet_threshold
                and distance_bifurcation > 0.05
                and previous_val * val[1] < 0
            ):
                coeff = np.real(previous_val / (previous_val - val[1]))
                traj_bifurcation = (
                    coeff * branch["traj"][-1] + (1 - coeff) * branch["traj"][-2]
                )
                dt_bifurcation = coeff * branch["dt"][-1] + (1 - coeff) * branch["dt"][-2]
                to_explore.append((traj_bifurcation, dt_bifurcation, vec[:, 0]))
                to_explore.append((traj_bifurcation, dt_bifurcation, vec[:, 1]))
                break

            dE = np.clip(0.1 * np.abs(val[1]), dE_min, dE_max)
            previous_val = val[1]

        return branch, to_explore, floquet_multipliers

    return (explore_component,)


@app.cell
def _(explore_component, n_steps, np, utils, x0):
    traj_eval, dt_eval = utils.init_trajectory(x0)
    branch, to_explore, multipliers = explore_component(
        traj_eval, dt_eval, np.array([1.0, 0.0, 0.0, 0.0]), n_steps
    )
    return branch, multipliers, to_explore


@app.cell
def _(branch, multipliers, np, plt):
    _traj_collection = np.array(branch["traj"])
    # print(repr(np.array(to_explore[1][0][:,0])))
    # print(repr(np.array(to_explore[1][1])))
    # print(repr(np.array(to_explore[1][2])))

    _fig = plt.figure(figsize=(10,5))
    _ax1 = _fig.add_subplot(1,2,1)
    _ax1.plot(np.array(branch["E"])[2:], multipliers, marker='.',linestyle ='')

    _ax2 = _fig.add_subplot(1,2,2,projection='3d')
    _ax2.plot(_traj_collection[:,1,:],_traj_collection[:,2,:],_traj_collection[:,3,:],linewidth=2)
    _ax2.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    _ax2.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    _ax2.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    plt.show()
    return


@app.cell
def _(np, plt, simple_continuation, to_explore):
    _idx = 1
    _branch, _multipliers = simple_continuation(
        to_explore[_idx][0],
        to_explore[_idx][1],
        to_explore[_idx][2],
        200,
        0.01
    )

    traj_collection = np.array(_branch["traj"])

    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(1,2,1)
    ax1.plot(np.array(_branch["E"])[2:], _multipliers, marker='.',linestyle ='')
    # ax1.set_ylim(-1,1)

    ax2 = fig.add_subplot(1,2,2,projection='3d')
    ax2.plot(traj_collection[::10,1,:],traj_collection[::10,2,:],traj_collection[::10,3,:],linewidth=2)
    ax2.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax2.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax2.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    plt.show()
    return


if __name__ == "__main__":
    app.run()
