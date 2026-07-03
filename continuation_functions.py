import casadi as cs
import numpy as np
import hyperparameters as params
import dynamics
import floquet


def energy_apex(x):
    # apex: dy = 0
    # x = [y,theta,dx,dtheta]^T
    return x[0] + 0.5*x[2]**2


def compute_delta(x0, dir, DE):
    if np.abs(dir[2]) <= 1e-6:
        # hopping in place case
        return DE / dir[0]
    a = dir[2] ** 2 / 2
    b = x0[2] * dir[2] + dir[0]
    c = -DE
    discr = b**2 - 4 * a * c
    d1, d2 = (-b - cs.sqrt(discr)) / (2 * a), (-b + cs.sqrt(discr)) / (2 * a)
    delta = d1 if np.abs(d1) < np.abs(d2) else d2
    return delta
