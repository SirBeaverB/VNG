from codes.data.io import load_dataset
from codes.pytorch_code.propagation import vng_mask_adj_matrix, vng_moving_nodes
import numpy as np
from codes.pytorch_code.propagation import vng_compute_P, vng_power_method


r = np.array([0.2, 0.3,0.4,0.1])
r=r.reshape((4,1))
P = vng_compute_P(0.1, np.array([[0, 0.5,0.5,0], [0.25, 0.25,0.25,0.25], [0.1, 0.2,0.3,0.4],[0.1,0.1,0.1,0.7]]), r)
print(P)
g=3
n = P.shape[0]
theta = r[g:] # (n-g)*1
e = np.ones((n-g, 1))  # (n-g)*1
s_T = theta.T/(theta.T @ e) # 1*(n-g)
print(s_T)
I = np.eye(g)

rows_I, cols_I = g, g
rows_e, cols_e = e.shape
rows_s_T, cols_s_T = s_T.shape

E = np.zeros((rows_I + rows_e, cols_I + cols_e))
E[:rows_I, :cols_I] = I
E[rows_I:, cols_I:] = e
print(E)

S = np.zeros((rows_I+rows_s_T, cols_I+cols_s_T))
S[:rows_I, :cols_I] = I
S[rows_I:, cols_I:] = s_T
print(S)

U = S @ P @ E
print(U)

P11 = P[:g, :g] # g*g
P12 = P[:g, g:] # g*(n-g)
P21 = P[g:, :g] # (n-g)*g
P22 = P[g:, g:] # (n-g)*(n-g)

U11 = P11 # g*g

e = np.ones((P12.shape[1], 1))  # (n-g)*1
U12 = P12 @ e

theta = r[g:] # (n-g)*1
s_T = theta.T/(theta.T @ e) # 1*(n-g)
U21 = s_T @ P21 # 1*g

e_22 = np.ones((g, 1)) # g*1
U22 = 1 - U21 @ e_22 # 1*1

top = np.hstack((U11, U12))
bottom = np.hstack((U21, U22))
U = np.vstack((top, bottom)) # (g+1)*(g+1)
print(U)