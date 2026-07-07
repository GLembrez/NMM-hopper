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

    import floquet 
    import dynamics
    import hyperparameters as params
    import continuation_functions as utils
    from continuation_opt import ContinuationSolver


    return ContinuationSolver, cs, eig, floquet, np, params, plt, utils


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
def _(K, W, np, params, plt, solver, utils):
    x0 = np.array([0.,1.,0.,0.,0.1,0.])
    xfh,xsh,tf,ts,y_a,t_a = utils.init_trajectory(x0,K,W)
    solver.initialize(ts/params.N_S,tf/params.N_F,xsh,xfh,utils.energy(x0)+0.01)
    xfh,xsh,dtf,dts = solver.solve()
    plt.plot(np.linspace(0,params.N_F*dtf,params.N_F),xfh[1,:])
    plt.plot(np.linspace(params.N_F*dtf,params.N_F*dtf + params.N_S*dts,params.N_S),xsh[1,:])
    return dtf, dts, xfh, xsh


@app.cell
def _(K, W, cs, dtf, dts, eig, floquet, params, xfh, xsh):
    var_eval = cs.vertcat(
        xfh[:,0], xsh[:,0], xfh[:,params.N_F//2], dtf,dts,dtf
    )
    x_0 = xfh[:,params.N_F//2]
    var_eval = floquet.full_newton(var_eval,x_0,K,W)
    J = floquet.var_newton(var_eval,x_0,K,W)
    R = floquet.residual_newton(var_eval,x_0,K,W)

    print(eig(J))

    return


@app.cell
def _():
    # flight_integrate = dynamics.RK4f.fold(params.N_F // 2)
    # stance_integrate = dynamics.R.1K4s.fold(params.N_S)

    # var = cs.SX.sym("var", 19)
    # x_init = cs.SX.sym("x_init", 6)


    # residual = cs.vertcat(
    #     var[:6] - flight_integrate(x_init,0.0, var[16], W),
    #     var[6:10] - stance_integrate(dynamics.flight_to_stance(var[:6]),0.0, var[17], K),
    #     var[10:16] - flight_integrate(dynamics.stance_to_flight(var[6:10]),0.0, var[18], W),
    #     cs.cos(var[2]) - var[1],
    #     var[6] ** 2 + var[7] ** 2 - 1,
    #     var[14],
    # )
    # newton = var - cs.inv(cs.jacobian(residual, var)) @ var

    # simple_shooting = cs.Function("simple_shooting", [var, x_init], [newton, residual])

    # initial_height = 1.1
    # var_eval = np.zeros((19,))
    # var_eval[1] = 1.0
    # var_eval[4] =  - np.sqrt(2*(initial_height - 1))
    # var_eval[7] = 1.0
    # var_eval[9] = np.sqrt(2*(initial_height - 1))
    # var_eval[11] = initial_height
    # var_eval[16] = 2*np.sqrt(2*(initial_height - 1)) / params.N_F
    # var_eval[17] = 0.1
    # var_eval[18] = 2*np.sqrt(2*(initial_height - 1)) / params.N_F


    # x0 = np.array([0.0, initial_height, 0.0, 0.0, 0.0, 0.0])
    # (var_eval, 
    # stance_integrate(dynamics.flight_to_stance(var_eval[:6]),0.0, var_eval[17], K), 
    # simple_shooting(var_eval, x0))
    return


if __name__ == "__main__":
    app.run()
