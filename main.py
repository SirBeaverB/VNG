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

NODE_PER_MASK = 50
N_MASKS = 10


if __name__ == '__main__':

    logging.basicConfig(
            format='%(asctime)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            level=logging.INFO)

    graph_name = 'cora_ml'  # - alternative dataset 'citeseer' and 'pubmed' - #
    graph = load_dataset(graph_name)
    graph.standardize(select_lcc=True) #找出最大联通子图

    nodes = list(range(NODE_PER_MASK*N_MASKS))
    graph_copy = copy.deepcopy(graph)
    #print("graph_copy1 = " + str(graph_copy.adj_matrix.shape))
    #print("graph1 = " + str(graph.adj_matrix.shape))    
    subgraph = create_subgraph(graph_copy, nodes_to_remove = nodes)
    #print("subgraph = " +str(subgraph.attr_matrix.shape))
    #print("graphcopy2 = " +str(graph_copy.adj_matrix.shape))
    #print("graph2 = "+str(graph.adj_matrix.shape))

    # - Train PPNP for the initial inputs for SDG - #
    start_time = time.time()

    prop_ppnp = PPRExact(subgraph.adj_matrix, alpha=0.1)
    # prop_appnp = PPRPowerIteration(graph.adj_matrix, alpha=0.1, niter=10)

    model_args = {
        'hiddenunits': [64], 
        'drop_prob': 0.5,    #drop掉50%的神经元
        'propagation': prop_ppnp}  # - alternative 'propagation': prop_appnp - #

    idx_split_args = {'ntrain_per_class': 20, 'nstopping': 500, 'nknown': 1500, 'seed': 2413340114} 
    reg_lambda = 5e-3
    learning_rate = 0.01

    test = False
    device = 'cpu'
    print_interval = 20

    model, result, Z = train_model(
            graph_name, agnostic_model, subgraph, model_args, learning_rate, reg_lambda, 
            idx_split_args, stopping_args, test, device, None, print_interval)

    print('Training PPNP costs: ' + str(time.time() - start_time) + ' sec.')

    # - SDG receives PPNP and fine-tunes on the updated graph - #
    start_time = time.time()

    """sdg = SDG(graph.adj_matrix, alpha=0.1).to(device)

    model_args = {
        'hiddenunits': [64],
        'drop_prob': 0.5,
        'propagation': sdg}
    
    model, result = fine_tune(
        graph_name, model, graph, model_args, learning_rate, reg_lambda,
        idx_split_args, stopping_args, test, device, None, print_interval)
    
    print('Generating the new graph + Training SDG costs: ' + str(time.time() - start_time) + ' sec.')"""

    # VNG

    nodes = list(range(NODE_PER_MASK*N_MASKS))
    graph_copy = copy.deepcopy(graph)
    #print("graph_copy1 = " + str(graph_copy.adj_matrix.shape))
    #print("graph1 = " + str(graph.adj_matrix.shape))    
    subgraph = create_subgraph(graph_copy, nodes_to_remove = nodes)
    #print("subgraph = " +str(subgraph.attr_matrix.shape))
    #print("graphcopy2 = " +str(graph_copy.adj_matrix.shape))
    #print("graph2 = "+str(graph.adj_matrix.shape))
    model, result, Z = train_model(
        graph_name, agnostic_model, subgraph, model_args, learning_rate, reg_lambda,   #这里有另外一个东西没有改
        idx_split_args, stopping_args, test, device, None, print_interval)

    adj_mat_list = vng_mask_adj_matrix(graph.adj_matrix, N_MASKS, NODE_PER_MASK)
    adj_mat_list.reverse()
    adj_attr_list = vng_mask_attr_matrix(graph.attr_matrix, N_MASKS, NODE_PER_MASK)
    adj_attr_list.reverse()
    if len(adj_mat_list) < 2:
        logging.error("adj_mat_list does not have enough elements to proceed.")
        exit(1)
    for i in range(1, len(adj_mat_list)):
        i_adj_matrix = adj_mat_list[i]
        i_attr_matrix = adj_attr_list[i]
        i_moved_adj_matrix, i_moved_attr_matrix, g = vng_moving_nodes(i_adj_matrix, i_attr_matrix, NODE_PER_MASK)
        vng = VNG(i_moved_adj_matrix, i_moved_attr_matrix, alpha=0.1, Z=Z, g=g).to(device)

        model_args = {
            'hiddenunits': [64],
            'drop_prob': 0.5,
            'propagation': vng}
        graph_new = copy.deepcopy(graph)
        nodes_to_remove = list(range(NODE_PER_MASK * (N_MASKS - i), NODE_PER_MASK * (N_MASKS - i + 1)))
        subgraph_new = create_subgraph(graph_new, nodes_to_remove = nodes_to_remove)
        model, result = fine_tune(
            graph_name, model, subgraph_new, model_args, learning_rate, reg_lambda,   #graph是新的图
            idx_split_args, stopping_args, test, device, None, print_interval)
        print('Generating the new graph + Training VNG costs: ' + str(time.time() - start_time) + ' sec.')
