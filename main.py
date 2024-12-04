import logging
import time

from codes.pytorch_code.agnostic_model import agnostic_model
from codes.pytorch_code.training import train_model
from codes.pytorch_code.training import fine_tune
from codes.pytorch_code.earlystopping import stopping_args
from codes.pytorch_code.propagation import PPRExact, PPRPowerIteration, SDG, VNG
from codes.pytorch_code.propagation import vng_mask_adj_matrix, vng_mask_attr_matrix, vng_moving_nodes
from codes.data.io import load_dataset
from codes.data.sparsegraph import create_subgraph
import copy
import torch

NODE_PER_MASK = 50
N_MASKS = 10
ALPHA = 0.1


if __name__ == '__main__':

    logging.basicConfig(
            format='%(asctime)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            level=logging.INFO)

    graph_name = 'movielens'  # 'cora_ml' 'citeseer'  'pubmed' 'ms_academic' 'movielens'- #
    graph = load_dataset(graph_name)
    graph.standardize(select_lcc=True)

    nodes = list(range(NODE_PER_MASK*N_MASKS))
    graph_copy = copy.deepcopy(graph)  
    subgraph = create_subgraph(graph_copy, nodes_to_remove = nodes)

    # - Train PPNP for the initial inputs for SDG - #
    start_time = time.time()

    #idx_split_args = {'ntrain_per_class': 20, 'nstopping': 500, 'nknown': 1500, 'seed': 2413340114} 
    idx_split_args = {'ntrain_per_class': 5, 'nstopping': 500, 'nknown': 1500, 'seed': 2413340114} #use this for ms_academic and movielens
    reg_lambda = 5e-3
    learning_rate = 0.01

    test = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print_interval = 200

    for i in range(N_MASKS):
        graph_new_ppnp = copy.deepcopy(graph)
        nodes_to_remove = list(range(NODE_PER_MASK * (N_MASKS - i - 1)))
        subgraph_new_ppnp = create_subgraph(graph_new_ppnp, nodes_to_remove = nodes_to_remove)
        #prop_ppnp = PPRExact(subgraph_new_ppnp.adj_matrix, alpha=ALPHA)
        start_time = time.time()
        prop_appnp = PPRPowerIteration(subgraph_new_ppnp.adj_matrix, alpha=ALPHA, niter=10)
        model_args = {
            'hiddenunits': [64], 
            'drop_prob': 0.5,    
            'propagation': prop_appnp}
        #'propagation': prop_ppnp} # - alternative 'propagation': prop_appnp - #
        model, result, Z = train_model(
            graph_name, agnostic_model, subgraph_new_ppnp, model_args, learning_rate, reg_lambda, 
            idx_split_args, stopping_args, test, device, None, print_interval)
    
        print('Training APPNP' + str(i)  + 'costs: ' + str(time.time() - start_time) + ' sec.')

    # VNG  

    adj_mat_list = vng_mask_adj_matrix(graph.adj_matrix, N_MASKS, NODE_PER_MASK)
    adj_mat_list.reverse()
    
    adj_attr_list = vng_mask_attr_matrix(graph.attr_matrix, N_MASKS, NODE_PER_MASK)
    adj_attr_list.reverse()

    graph_new = copy.deepcopy(graph)
    nodes_to_remove = list(range(NODE_PER_MASK * (N_MASKS - 1)))
    subgraph_new = create_subgraph(graph_new, nodes_to_remove = nodes_to_remove)

    # i = 0
    #prop_ppnp = PPRExact(subgraph_new.adj_matrix, alpha=ALPHA)
    start_time = time.time()
    prop_appnp = PPRPowerIteration(subgraph_new.adj_matrix, alpha=ALPHA, niter=10)
    model_args = {
        'hiddenunits': [64], 
        'drop_prob': 0.5,    
        'propagation': prop_appnp} # - alternative 'propagation': prop_appnp - #
    model, result, Z = train_model(
            graph_name, agnostic_model, subgraph_new, model_args, learning_rate, reg_lambda, 
            idx_split_args, stopping_args, test, device, None, print_interval)
    print('Training basic graph for VNG costs: ' + str(time.time() - start_time) + ' sec.')
    

    if len(adj_mat_list) < 2:
        logging.error("adj_mat_list does not have enough elements to proceed.")
        exit(1)
    for i in range(1, len(adj_mat_list) - 1):
        i_adj_matrix = adj_mat_list[i-1]
        i_attr_matrix = adj_attr_list[i-1]
        i_moved_adj_matrix, i_moved_attr_matrix, g = vng_moving_nodes(i_adj_matrix, i_attr_matrix, NODE_PER_MASK)
        
        graph_new = copy.deepcopy(graph)
        nodes_to_remove = list(range(NODE_PER_MASK * (N_MASKS - i - 1)))
        subgraph_new = create_subgraph(graph_new, nodes_to_remove = nodes_to_remove)
        start_time = time.time()
        vng = VNG(subgraph_new.adj_matrix, alpha=ALPHA, niter=10, old_Z=Z, g=g).to(device)
        model_args = {
            'hiddenunits': [64],
            'drop_prob': 0.5,
            'propagation': vng}
        
        model, result, Z = fine_tune(
            graph_name, model, subgraph_new, model_args, learning_rate, reg_lambda,   #subgraph_new is new
            idx_split_args, stopping_args, test, device, None, print_interval)
        print('Generating the new graph ' + str(i) + ' Training VNG costs: ' + str(time.time() - start_time) + ' sec.')
