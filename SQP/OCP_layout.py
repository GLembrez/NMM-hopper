import casadi as cs
import numpy as np

class Layout:
    def __init__(self, modes, N, dim):
        self.modes = tuple(modes)
        self.N = N
        self.dim = dim
        self.dim_terminal = dim[modes[-1]] - int(modes[-1] == "f")
        self.dim_terminal_aux = 2 + 2*self.dim_terminal # E, phi, s+, s-
        # stages = (mode_index, mode_name, node_index)
        self.stages = [(i,m,k) for i,m in enumerate(modes) for k in range(N[m])] 
        self.N_ocp = len(self.stages) - 1
        self.edge = []      # edge[k] = "flow"/"reset"/"terminal" 
        self.nx = []        # nx[k] = dimension of state at stage k
        self.nu = []        # nu[k] = dimension of control at stage k

        for k, (i, m, _) in enumerate(self.stages):
            final = k == self.N_ocp
            # x_aug = [x tau t]^T if not terminal 
            # x_aug = [x tau t E phi s+ s-] if terminal
            self.nx.append(dim[m] + 2 + final*self.dim_terminal_aux)
            if final:
                self.edge.append("terminal")
                self.nu.append(0)
            elif self.stages[k+1][0] != i:
                self.edge.append("reset")
                self.nu.append(1) # the control is the next mode duration
            else:
                self.edge.append("flow")
                # control of size 1 except just before the terminal state: 
                # v_(N+1) = [u_(N-1) E phi s+ s-]
                self.nu.append(1+(k==self.N_ocp-1) * self.dim_terminal_aux)

        offset = 0 # current position in the flattened QP vector
        self.x_slice, self.u_slice = [], []
        for nx,nu in zip(self.nx, self.nu):
            self.x_slice.append(slice(offset, offset+nx))
            offset += nx
            self.u_slice.append(slice(offset, offset+nu))
            offset += nu
        self.nw = offset # size of w = [x0,u0,x1,u1 ... xN]
        # find [E, phi, s_plus, s_minus] in terminal x and u 
        self.x_t_slice = slice(self.x_slice[-1].stop - self.dim_terminal_aux, self.x_slice[-1].stop)
        self.u_t_slice = slice(self.u_slice[-2].start+1, self.u_slice[-2].stop)

    def split(self,w):
        states = [w[s] for s in self.x_slice]
        controls = [w[s] for s in self.u_slice]
        return states, controls

    def state_parts(self, states, k):
        # returns (x,tau,t)
        d = self.dim[self.stages[k][1]]
        return states[k][:d], states[k][d], states[k][d+1]

    def pack(self, x, u, duration, E, phi): 
        w = []
        mode_start = [cs.sum1(duration[:i]) for i in range(len(self.modes))]
        x_t_aux = cs.vertcat(E, phi, cs.DM.zeros(2*self.dim_terminal))
        for k,(i,_,j) in enumerate(self.stages):
            # compute w_k = [... y_k v_k ...]
            y_k = cs.vertcat(x[i][:, j], duration[i], mode_start[i])
            w.append(cs.vertcat(y_k, x_t_aux) if k == self.N_ocp else y_k)
            if self.edge[k] == "flow":
                # v_k = u_ij
                w.append(cs.vertcat(u[i][j], x_t_aux) if k == self.N_ocp-1 else u[i][j])
            elif self.edge[k] == "reset":
                # v_k = tau_(i+1)
                w.append(duration[i+1])
        return cs.vertcat(*w)

    def unpack(self,w):
        y,v = self.split(w)
        x = [np.zeros((self.dim[m],self.N[m])) for m in self.modes]
        u = [np.zeros(self.N[m]-1) for m in self.modes]
        t = np.zeros(len(self.modes))

        for k,(idx_m,m,idx_n) in enumerate(self.stages):
            d = self.dim[m]
            x[idx_m][:,idx_n] = np.asarray(y[k][:d]).squeeze()
            if idx_n == 0:
                t[idx_m] = float(y[k][d]) 
            if self.edge[k] == "flow":
                u[idx_m][idx_n] = float(v[k][0])
        terminal_aux = np.asarray(y[-1][-self.dim_terminal_aux:]).squeeze()

        return {
            "modes": self.modes,
            "x": x,
            "u": u,
            "t": t.tolist(),
            "E": float(terminal_aux[0]),
            "phi": float(terminal_aux[1]),
            "s_plus": terminal_aux[2:2 + self.dim_terminal],
            "s_minus": terminal_aux[2 + self.dim_terminal:],
        }
