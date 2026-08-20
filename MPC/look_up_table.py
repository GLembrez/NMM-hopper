import casadi as cs
import numpy as np
import pickle
from matplotlib import pyplot as plt

N_SAMPLES = 100

data_locomotion = pickle.load(open('data/data_test.pkl', 'rb'))
NMM = np.array(data_locomotion[2])

N = 20000
skip = max(1,int(N/N_SAMPLES))
NMM_sampled = NMM[:N:skip,:]


LUT_x = cs.interpolant('LUTx','bspline',[NMM_sampled[:,-1]],NMM_sampled[:,0])
LUT_y = cs.interpolant('LUTy','bspline',[NMM_sampled[:,-1]],NMM_sampled[:,1])
LUT_dx = cs.interpolant('LUTdx','bspline',[NMM_sampled[:,-1]],NMM_sampled[:,3])
LUT_dy = cs.interpolant('LUTdy','bspline',[NMM_sampled[:,-1]],NMM_sampled[:,4])
E_linspace = np.linspace(np.min(NMM_sampled[:,-1]),np.max(NMM_sampled[:,-1]),1000)


fig = plt.figure()
plt.plot(NMM[:,1],NMM[:,3])
plt.plot(NMM_sampled[:,1],NMM_sampled[:,3])
plt.plot(LUT_y(E_linspace),LUT_dx(E_linspace))
plt.show()
