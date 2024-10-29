import numpy as np
nodes_per_mask = 2

new_adj_matrix = np.matrix([[0, 0.5, 0.2, 0.1], [0.25, 0.25, 0.25, 0.25], [0.1, 0.2, 0.3, 0.4], [0.1, 0.1, 0.1, 0.7]])


influenced_nodes = set()
neighbors = new_adj_matrix[0].nonzero()[1]
print(neighbors)
neighbors = [n for n in neighbors if n >= nodes_per_mask]  # Remove neighbors with indice less than nodes_per_mask
print(neighbors)
influenced_nodes.update(neighbors)
print(influenced_nodes)