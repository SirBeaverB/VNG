import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import math
import random
import time
import torch.nn.functional as F

from .utils import MixedDropout, sparse_matrix_to_torch

def calc_A_hat(adj_matrix: sp.spmatrix) -> sp.spmatrix:
    nnodes = adj_matrix.shape[0]
    A = adj_matrix + sp.eye(nnodes) # add self-loop
    D_vec = np.sum(A, axis=1).A1 # degree
    D_vec_invsqrt_corr = 1 / D_vec #D^(-1) 
    D_invsqrt_corr = sp.diags(D_vec_invsqrt_corr) # vector to diagonal matrix
    return A @ D_invsqrt_corr #A_hat = A @ D^(-1)

def calc_ppr_exact(adj_matrix: sp.spmatrix, alpha: float) -> torch.Tensor:
    nnodes = adj_matrix.shape[0]
    M = calc_A_hat(adj_matrix)
    A_inner = sp.eye(nnodes) - (1 - alpha) * M
    A_inner = torch.tensor(A_inner.toarray(), dtype=torch.float32)
    return alpha * torch.linalg.inv(A_inner)


def mask_adj_matrix(adj_matrix: sp.spmatrix) -> sp.spmatrix:
    nnodes = adj_matrix.shape[0]
    masked_n_edges = 100
    influenced_n_nodes = int(math.sqrt(masked_n_edges / 2))
    influenced_nodes = random.sample(range(1, nnodes), influenced_n_nodes)
    masked_adj_matrix = adj_matrix.copy()
    for i in range(len(influenced_nodes)):
        for j in range(i + 1, len(influenced_nodes)):
            masked_adj_matrix[influenced_nodes[i], influenced_nodes[j]] = 0
            masked_adj_matrix[influenced_nodes[j], influenced_nodes[i]] = 0
    return masked_adj_matrix

def vng_mask_adj_matrix(adj_matrix: sp.spmatrix, n_masks: int, nodes_per_mask: int) -> list:
    """
    Splits the adjacency matrix adj_matrix by masking a fixed number of nodes from the top in each iteration, 
    generating progressively smaller graphs.
    
    Parameters:
    - adj_matrix: The original adjacency matrix of the graph.
    - n_masks: The number of iterations (each time masking a fixed number of nodes).
    - nodes_per_mask: The number of nodes to mask in each iteration, starting from the top of the matrix.
    
    Returns:
    - results: A list containing the adjacency matrix after each node mask.
    """
    results = []
    current_adj_matrix = adj_matrix.copy()
    results.append(current_adj_matrix)
    nnodes = current_adj_matrix.shape[0]
    for _ in range(n_masks):
        if nnodes <= nodes_per_mask:
            break
        mask_adj_matrix = current_adj_matrix[nodes_per_mask:, nodes_per_mask:]
        nnodes -= nodes_per_mask       
        results.append(mask_adj_matrix)
        current_adj_matrix = mask_adj_matrix
    return results

def vng_mask_attr_matrix(attr_matrix: sp.spmatrix, n_masks: int, nodes_per_mask: int) -> list:
    """
    Splits the attribute matrix attr_matrix by masking a fixed number of nodes from the top in each iteration.
    
    Parameters:
    - adj_matrix: The original attribute matrix of the graph.
    - n_masks: The number of iterations (each time masking a fixed number of nodes).
    - nodes_per_mask: The number of nodes to mask in each iteration, starting from the top of the matrix.
    
    Returns:
    - results: A list containing the attribute matrix after each node mask.
    """
    results = []
    current_attr_matrix = attr_matrix.copy()
    results.append(current_attr_matrix)
    nnodes = current_attr_matrix.shape[0]

    for _ in range(n_masks):
        if nnodes <= nodes_per_mask:
            break
        mask_attr_matrix = current_attr_matrix[nodes_per_mask:, :]
        nnodes -= nodes_per_mask       
        results.append(mask_attr_matrix)
        current_attr_matrix = mask_attr_matrix
    
    return results

def vng_moving_nodes(new_adj_matrix: sp.spmatrix, new_attr_matrix: sp.spmatrix, nodes_per_mask: int) -> tuple:
    """
    Rearranges the adjacency matrix of the new graph by moving all changed nodes 
    (newly added nodes and nodes connected to them) to the top-left corner.
    
    Parameters:
    - new_adj_matrix: The adjacency matrix of the new graph, where new nodes are already at the top-left.
    - new_attr_matrix: The attribute matrix of the new graph, where new nodes are already at the top.
    - nodes_per_mask: The number of newly added nodes, already located in the top-left of the new adjacency matrix.
    
    Returns:
    - adjusted_adj_matrix: The new adjacency matrix with changed nodes (new nodes and their neighbors) moved to the top-left corner.
    - num_changed_nodes: The number of changed nodes moved to the top-left corner.
    """
    new_nodes = set(range(nodes_per_mask))
    influenced_nodes = new_nodes.copy()
    for node in new_nodes:
        neighbors = new_adj_matrix[node].nonzero()[1]
        influenced_nodes.update(neighbors)

    influenced_nodes = sorted(influenced_nodes)
    remaining_nodes = [node for node in range(new_adj_matrix.shape[0]) if node not in influenced_nodes]
    
    new_order = influenced_nodes + remaining_nodes
    adjusted_adj_matrix = new_adj_matrix[new_order, :][:, new_order]
    adjusted_attr_matrix = new_attr_matrix[new_order, :]
    
    num_changed_nodes = len(influenced_nodes)
    
    return adjusted_adj_matrix, adjusted_attr_matrix, num_changed_nodes



def track_ppr(adj_matrix: sp.spmatrix, masked_adj_matrix: sp.spmatrix, ppr_mat, alpha):
    nnodes = adj_matrix.shape[0]

    A = masked_adj_matrix + sp.eye(nnodes)
    D_vec = torch.sum(torch.tensor(A.toarray()), dim=1)
    D_vec_invsqrt_corr = 1 / torch.sqrt(D_vec)
    D_invsqrt_corr = sp.diags(D_vec_invsqrt_corr)
    M = A @ D_invsqrt_corr

    A_prime = adj_matrix + sp.eye(nnodes)
    D_vec_prime = torch.sum(torch.tensor(A_prime.toarray()), dim=1)
    D_vec_invsqrt_corr_prime = 1 / torch.sqrt(D_vec_prime)
    D_invsqrt_corr_prime = sp.diags(D_vec_invsqrt_corr_prime)
    M_prime = A_prime @ D_invsqrt_corr_prime

    # --- push to approximate converged --- #
    diff_matrix = M_prime - M
    pushout = alpha * diff_matrix @ ppr_mat.T  # - probability mass that needs to be pushed out - #
    acc_pushout = pushout                      # k = 0

    temp = alpha * M_prime                     # k = 1
    acc_pushout += temp @ pushout

    num_itr = 1                                # k starts from 2 to user-specified
    for k in range(num_itr):
        new_temp = temp * alpha @ M_prime
        acc_pushout += new_temp @ pushout
        temp = new_temp

    t_ppr = ppr_mat + acc_pushout.T
    # ------------------------------------ #

    return t_ppr

def vng_compute_P(alpha, A, r): 
    A_T = torch.tensor(A.toarray(), dtype=torch.float32).T
    e = torch.ones((A.shape[0], 1), dtype=torch.float32)
    P = (1 - alpha) * A_T + alpha * e@(r.T)
    return P

def vng_power_method(U, tol=1e-8, max_iter=5):
    U = U / U.sum(axis=1)#.reshape(-1, 1)
    n = U.shape[0]
    phi_T = torch.ones(n, dtype=torch.float32) / n
    for _ in range(max_iter):
        phi_T_next = phi_T @ U
        if torch.norm(phi_T_next - phi_T, p=1) < tol:
            break
        phi_T = phi_T_next
    return phi_T_next

def vng_track_pi(new_adj_matrix: sp.spmatrix, old_adj_matrix: sp.spmatrix, alpha, r, g):
    M = calc_A_hat(old_adj_matrix)
    M_prime = calc_A_hat(new_adj_matrix)

    # step 1
    #calculate P
    P = vng_compute_P(alpha, M_prime, r) #r should be n*1

    n = P.shape[0]

    """P11 = P[:g, :g] # g*g
    P12 = P[:g, g:] # g*(n-g)
    P21 = P[g:, :g] # (n-g)*g
    P22 = P[g:, g:] # (n-g)*(n-g)"""

    # initialize s_T
    theta = r[g:] # (n-g)*1
    e = torch.ones((n-g, 1), dtype=torch.float32)  # (n-g)*1
    s_T = theta.T/(theta.T @ e) # 1*(n-g)
    I = torch.eye(g, dtype=torch.float32)

    rows_I, cols_I = g, g
    rows_e, cols_e = e.shape
    rows_s_T, cols_s_T = s_T.shape

    E = torch.zeros((rows_I + rows_e, cols_I + cols_e), dtype=torch.float32)
    E[:rows_I, :cols_I] = I
    E[rows_I:, cols_I:] = e

    S = torch.zeros((rows_I + rows_s_T, cols_I + cols_s_T), dtype=torch.float32)
    S[:rows_I, :cols_I] = I
    S[rows_I:, cols_I:] = s_T

    for _ in range(1):
        # step 2
        """U11 = P11 # g*g

        e = np.ones((P12.shape[1], 1))  # (n-g)*1
        U12 = P12 @ e

        theta = r[g:] # (n-g)*1
        s_T = theta.T/(theta.T @ e) # 1*(n-g)
        U21 = s_T @ P21 # 1*g

        e_22 = np.ones((g, 1)) # g*1
        U22 = 1 - U21 @ e_22 # 1*1

        top = np.hstack((U11, U12))
        bottom = np.hstack((U21, U22))
        U = np.vstack((top, bottom)) # (g+1)*(g+1)"""
        

        U = S @ P @ E
        
        # step 3
        phi_T = vng_power_method(U) # (g+1)*1

        # step 4
        phi_g = phi_T[:g]
        phi_s = phi_T[g] * s_T
        phi_g = phi_g.unsqueeze(0)  
        pi = torch.cat((phi_g, phi_s), dim=1) 

        pi_hat_T = pi @ P #T?
        if torch.norm(pi_hat_T - pi, p=1) < 0.0001:
            break

        theta = pi_hat_T[:, g:] # 1*(n-g)
        s_T = theta/(theta @ e) # 1*(n-g)

        S = torch.zeros((rows_I + rows_s_T, cols_I + cols_s_T), dtype=torch.float32)
        S[:rows_I, :cols_I] = I
        S[rows_I:, cols_I:] = s_T

    return pi

#PPNP
class PPRExact(nn.Module):    
    def __init__(self, adj_matrix: sp.spmatrix, alpha: float, drop_prob: float = None):
        super().__init__()

        ppr_mat = calc_ppr_exact(adj_matrix, alpha) 
        self.register_buffer('mat', torch.FloatTensor(ppr_mat))

        if drop_prob is None or drop_prob == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(drop_prob)

    def forward(self, predictions: torch.FloatTensor, idx: torch.LongTensor):
        return self.dropout(self.mat[idx]) @ predictions   # - aggregating neighbourhood predictions - #

#APPNP
class PPRPowerIteration(nn.Module):   
    def __init__(self, adj_matrix: sp.spmatrix, alpha: float, niter: int, drop_prob: float = None):
        super().__init__()
        self.alpha = alpha
        self.niter = niter

        M = calc_A_hat(adj_matrix) #A normalized before adding self-loop
        self.register_buffer('A_hat', sparse_matrix_to_torch((1 - alpha) * M)) #A_hat = (1-alpha)M，存入buffer

        if drop_prob is None or drop_prob == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(drop_prob)

    def forward(self, local_preds: torch.FloatTensor, idx: torch.LongTensor):
        preds = local_preds #h0
        for _ in range(self.niter):
            A_drop = self.dropout(self.A_hat)
            preds = A_drop @ preds + self.alpha * local_preds
        return preds[idx] #返回idx对应的部分。
    


class SDG(nn.Module):
    def __init__(self, adj_matrix: sp.spmatrix, alpha: float, drop_prob: float = None):
        super().__init__()

        start_time = time.time()

        # last time graph structure and its ppr matrix
        masked_adj_matrix = mask_adj_matrix(adj_matrix)
        ppr_mat = calc_ppr_exact(masked_adj_matrix, alpha)

        print('Generating the new graph costs: ' + str(time.time() - start_time) + ' sec.')

        # tracked ppr matrix
        t_ppr = track_ppr(adj_matrix, masked_adj_matrix, ppr_mat, alpha)

        self.register_buffer('mat', torch.FloatTensor(t_ppr))

        if drop_prob is None or drop_prob == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(drop_prob)

    def forward(self, predictions: torch.FloatTensor, idx: torch.LongTensor):
        return self.dropout(self.mat[idx]) @ predictions

class VNG(nn.Module):
    def __init__(self, new_adj_matrix: sp.spmatrix,
                 old_Z, alpha: float, niter: int, g, drop_prob: float = None):
        super().__init__()

        #start_time = time.time()

        # last graph structure and its ppr matrix
        #self.adj_matrix = adj_matrix
        #self.attr_matrix = attr_matrix
        #ppr_mat = calc_ppr_exact(old_adj_matrix, alpha)
        
        #print('Generating the new graph costs: ' + str(time.time() - start_time) + ' sec.')

        # tracked pi matrix
        #columns = []
        rows = []
        n_new = new_adj_matrix.shape[0]
        n_old = old_Z.shape[0]
        n_delta = n_new - n_old
        for i in range(old_Z.shape[1]):
            r = torch.zeros((n_new, 1), dtype=torch.float32)
            r[n_delta:, 0] = old_Z[:, i] #add the zeros to the old Z, (n*1)
            t_pi = vng_track_pi(new_adj_matrix, new_adj_matrix, alpha, r, g) 
            rows.append(t_pi)
        pi_mat = torch.row_stack(rows) # n*k
        self.register_buffer('pi_mat', torch.FloatTensor(pi_mat))

        #self.register_buffer('mat', torch.FloatTensor(ppr_mat))

        if drop_prob is None or drop_prob == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(drop_prob)

        #
        self.alpha = alpha
        self.niter = niter

        M = calc_A_hat(new_adj_matrix) #A normalized before adding self-loop
        self.register_buffer('A_hat', sparse_matrix_to_torch((1 - alpha) * M)) #A_hat = (1-alpha)M，存入buffer

        if drop_prob is None or drop_prob == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(drop_prob)

    def forward(self, local_preds: torch.FloatTensor, idx: torch.LongTensor):
        preds = self.pi_mat.T
        for _ in range(self.niter):
            A_drop = self.dropout(self.A_hat)
            preds = A_drop @ preds + self.alpha * local_preds  #local_preds = h0
        return preds[idx] #return the part of idx