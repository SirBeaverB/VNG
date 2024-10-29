from sdg.data.io import load_dataset
from sdg.pytorch_code.propagation import vng_mask_adj_matrix, vng_moving_nodes
import numpy as np
from sdg.pytorch_code.propagation import vng_compute_P, vng_power_method


r = np.array([0.2, 0.3,0.4,0.1])
r=r.reshape((4,1))
P = vng_compute_P(0.1, np.array([[0, 0.5,0.5,0], [0.25, 0.25,0.25,0.25], [0.1, 0.2,0.3,0.4],[0.1,0.1,0.1,0.7]]), r)
print(P)
g=2
P11 = P[:g, :g]
P12 = P[:g, g:]
P21 = P[g:, :g]
P22 = P[g:, g:]
print(P11)
print(P12)
print(P21)
print(P22)
e = np.ones((P12.shape[1], 1))
print(e)
U11 = P11
U12 = P12 @ e
print(U12)
theta = r[g:]
print(theta)
s = theta.T/(theta.T @ e)
print(s)
U21 = s @ P21
print(U21)
e_22 = np.ones((g, 1)) # g*1
U22 = 1 - U21 @ e_22 # 1*1
print(U22)
top = np.hstack((U11, U12))
bottom = np.hstack((U21, U22))
U = np.vstack((top, bottom))
print (U)
phi_T = vng_power_method(U) # (g+1)*1
print(phi_T)
phi_g = phi_T[:g] # g*1
phi_s = (phi_T[g]*s)[0]
print(phi_g)
print(phi_s)
pi_T = np.concatenate((phi_g, phi_s))
print(pi_T)