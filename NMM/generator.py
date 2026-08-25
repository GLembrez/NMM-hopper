import casadi as cs
import numpy as np
from copy import deepcopy
from scipy.linalg import null_space

def find_regular_point(BP):
    u_star, p1, p2 = BP
    u1 = u_star + 0.05 * p1.reshape((11,))
    u2 = u_star + 0.05 * p2.reshape((11,))
    return u1,u2 

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

def check_diverging_newton(solver,u,d,h,p):
    diverging = True
    while diverging and h>1e-3:
        u_est = deepcopy(u) + d*h*p.reshape((11,))
        u_est_newton = cs.vertcat(solver.newton(u_est[:10],u_est[10]),u_est[10])
        err_newton = np.linalg.norm(solver.residual(u_est_newton))
        err = np.linalg.norm(solver.residual(u_est))
        if err_newton<err:
            diverging = False
            break
        h = h/2
    return diverging,h

def compute(solver,u0):

    M,BP,TP,IP = [],[],[],[]
    p_stored = None
    det_stored = None
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
        if det < 0:
            # set p to the forward direction
            p = - p

        # corrector step
        diverging,h = check_diverging_newton(solver,u,d,h,p)
        u += d * h * p.reshape((11,))
        if not diverging:
            solver.initialize(u)
            u = solver.solve()  
            h = solver.STEP_SIZE

        # handle the crossing of special points
        isSpecialPoint = True 
        if (u[1] <= 1.01 and np.abs(u[3]) <= 1e-6) :  
            print("unfeasable at step {}".format(step))
            IP.append(deepcopy(M[-1]))
        elif step > 2 and h<1e-3:
            print("diverging newton at step {}".format(step))
            IP.append(deepcopy(M[-1]))
        elif step > 2 and np.linalg.norm(M[-1]-u) > 100 * h:
            u = M[-1].copy()
            print("jump at step {}".format(step))
            IP.append(deepcopy(M[-1]))
        elif step > 2 and (p_stored.T @ p) < 0:
            print("bifurcation at step {}".format(step))
            coeff = det_stored / (det_stored + np.abs(det))
            u_star = coeff * M[-1] + (1-coeff) * M[-2]
            p1 = coeff * p - (1-coeff) * p_stored
            p1 = p1/np.linalg.norm(p1)
            p2 = process_bifurcation(u_star, p1, solver)
            BP.append((u_star,(M[-2] - M[-1])/np.linalg.norm(M[-2] - M[-1]),p2))
        # elif step>2 and np.linalg.det(solver.compute_Jz(u[:10],u[10]))*np.linalg.det(solver.compute_Jz(M[-1][:10],M[-1][10])) < 0:
        #     print("turning point at step {}".format(step))
        #     TP.append((u,p))
        elif step==solver.N_STEPS-1:
            print("max branch depth reached")
        else:
            isSpecialPoint = False 

        step += 1
        M.append(u)
        p_stored = p.copy()
        det_stored = np.abs(det)

        if isSpecialPoint:
            if d==1:
                step = 0
                d=-1
                u = u0.copy()
            else:
                break

    return M,BP,IP,TP
