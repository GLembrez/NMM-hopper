import casadi as cs
import numpy as np
from copy import deepcopy
from scipy.linalg import null_space



def process_bifurcation(u, p1, solver):

    J = solver.compute_J(u)
    val, vec = np.linalg.eig(cs.vertcat(J, p1.T))
    idx_val = np.argmin(np.abs(val))
    p2 = vec[:, idx_val].real.reshape((11,1))
    val, vec = np.linalg.eig(J @ J.T)
    idx_val = np.argmin(np.abs(val))
    e = vec[:, idx_val].real
    b12 = solver.hessian(u, p1, p2, e)
    b22 = solver.hessian(u, p2, p2, e)
    beta2 = 1
    beta1 = -b22 / (2 * b12)
    p2 = beta1 * p1 + beta2 * p2
    p2 = p2 / np.linalg.norm(p2)
    return np.array(p2)

def check_impossible(step,u,failed,M,h):
    physics = (u[1] <= 1.01 and np.abs(u[3]) <= 1e-6)
    jump = (step > 2 and np.linalg.norm(M[-1]-u) > 10 * h)
    if physics or jump:
        return True
    return failed


def compute(solver,u0):

    M,BP,TP,IP = [],[],[],[]
    p_stored = None
    det_stored = None
    detz_stored = None
    u = u0.copy()
    h = solver.STEP_SIZE
    M.append(u0)
    d = 1
    step = 0 

    while step<solver.N_STEPS:

        # predictor step
        J = solver.compute_J(u)
        p = null_space(J)
        det = np.linalg.det(cs.vertcat(J,p.T))
        detz = np.linalg.det(solver.compute_Jz(u[:10],u[10]))
        if det < 0:
            # set p to the forward direction
            p = - p

        # corrector step
        solverFailed = False
        u += d * h * p.reshape((11,))
        solver.initialize(u)
        try:
            u = solver.solve()  
        except:
            solverFailed = True

        # handle the crossing of special points
        isSpecialPoint = True 
        if  check_impossible(step,u,solverFailed,M,h):
            u = M[-1].copy()
            print("impossible at step {}".format(step))
            IP.append(deepcopy(M[-1]))
        elif step > 2 and (p_stored.T @ p) < 0:
            print("bifurcation at step {}".format(step))
            coeff = det_stored / (det_stored + np.abs(det))
            u_star = coeff * M[-1] + (1-coeff) * M[-2]
            p1 = coeff * p - (1-coeff) * p_stored
            p1 = p1/np.linalg.norm(p1)
            p2 = process_bifurcation(u_star, p1, solver)
            BP.append((u_star,-(M[-2] - M[-1])/np.linalg.norm(M[-2] - M[-1]),p2))
        # elif step>2 and detz*detz_stored < 0:
        #     print("turning point at step {}".format(step))
        #     coeff = np.abs(det_stored / (det_stored - det))
        #     u_star = coeff * u + (1-coeff) * M[-1] 
        #     # p1 = coeff * p + (1-coeff) * p_stored
        #     TP.append((u_star,p))
        elif step==solver.N_STEPS-1:
            print("max branch depth reached")
        else:
            isSpecialPoint = False 

        step += 1
        M.append(deepcopy(u))
        p_stored = p.copy()
        det_stored = np.abs(det)
        detz_stored = detz.copy()


        if isSpecialPoint:
            if d==1:
                step = 0
                d=-1
                u = u0.copy()
            else:
                break

    return M,BP,IP,TP
