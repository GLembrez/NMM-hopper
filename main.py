import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import casadi as cs
    from matplotlib import pyplot as plt

    import floquet 
    import dynamics
    import hyperparameters as params
    from continuation_opt import ContinuationSolver

    return ContinuationSolver, cs


@app.cell
def _(ContinuationSolver, cs):
    K = 40
    W = cs.sqrt(10)

    solver = ContinuationSolver(K,W)

    return


if __name__ == "__main__":
    app.run()
