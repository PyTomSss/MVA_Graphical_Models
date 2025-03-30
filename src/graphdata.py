import copy
import torch
import networkx as nx
import numpy as np
import os 
import random
import matplotlib.pyplot as plt
import subprocess
import sys

# Vérifier si le module 'node2vec' est installé

try:
    import node2vec
except ImportError:
    # Si le module n'est pas trouvé, essayer de l'installer via pip
    print("Le module 'node2vec' n'est pas installé. Installation en cours...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "node2vec"])
    import node2vec  # Réessayer d'importer après l'installation

# Maintenant, tu peux utiliser node2vec sans souci
from node2vec import Node2Vec


def get_node_random_walk(x_list, adj_list, node2vec_hidden, walk_length, num_walk, p, q, workers):
    """
    Generates node random walks using the Node2Vec algorithm for a given adjacency list.
    
    Parameters:
    - x_list: List of node features represented as one-hot vectors.
    - adj_list: List of adjacency matrices representing graphs.
    - node2vec_hidden: Dimension size for the Node2Vec embeddings.
    - walk_length: Number of steps in each random walk.
    - num_walk: Number of random walks per node.
    - p: Return parameter controlling the likelihood of returning to the previous node.
    - q: In-out parameter controlling the exploration tendency.
    - workers: Number of parallel workers for processing.
    
    Returns:
    - node_random_walk_list: A list of processed random walks as node features.
    """
    node_random_walk_list = []
    
    print('walk_length:', walk_length)
    print('num_walk:', num_walk)
    print('p:', p)
    print('q:', q)
    
    for i, adj in enumerate(adj_list):
        
        # Print progress every 15 iterations to track processing
        if i % 15 == 0:
            print('node random walk ...', i, '/', len(adj_list))
            
        walk_dic = {}  # Dictionary to store random walks for each node
        
        # Convert adjacency matrix to a NetworkX graph
        if type(adj).__module__ == np.__name__:
            G = nx.Graph(adj)  # Directly convert numpy array to a graph
        else:
            G = nx.Graph(adj.to('cpu').numpy())  # Convert PyTorch tensor to numpy, then to a graph
        
        # Initialize Node2Vec model with given parameters
        node2vec = Node2Vec(
            graph=G,  # The input graph must be a NetworkX graph
            dimensions=node2vec_hidden,  # Number of dimensions for embedding
            walk_length=walk_length,  # Number of nodes visited per walk
            num_walks=num_walk,  # Number of walks generated per node
            p=p,  # Return parameter: higher values make walks stay closer to the start node
            q=q,  # Exploration parameter: higher values encourage moving further away
            weight_key=None,  # Assume unweighted edges
            workers=workers,  # Number of parallel processing threads
            quiet=True  # Suppress detailed logs for cleaner output
        )

        # Process generated random walks and store them by node ID
        for random_walk in node2vec.walks:
            start_node = int(random_walk[0])  # Extract the starting node of the walk
            if start_node not in walk_dic:
                walk_dic[start_node] = []  # Initialize list for storing walks
            walk_dic[start_node].append(random_walk)  # Append the generated walk
        
        # Extract the indices of active features (1.0 values) from one-hot encoded vectors
        if type(x_list[i]).__module__ == np.__name__:
            hot_index = np.where(x_list[i] == 1.0)[1]
        else:
            hot_index = np.where(x_list[i].to('cpu').numpy() == 1.0)[1]
         
        # Convert random walks into node feature representations
        node_random_walk_list2 = []
        
        for a in range(len(adj)):
            walks = walk_dic[a]  # Retrieve all walks starting from node 'a'
            walks_list = []  # Initialize list to store processed walks
            
            for walk in walks:
                walk2 = []  # Temporary list to store transformed walk
                for node in walk:
                    node_id = int(node)
                    # Ensure the node index is within the valid range of one-hot indices
                    if node_id < len(hot_index):
                        walk2.append(float(hot_index[node_id]))
                
                # Pad walks with zeros if they are shorter than the defined walk_length
                walks_list.append([0.0] * (walk_length - len(walk2)) + walk2)

            node_random_walk_list2.append(np.array(walks_list))  # Convert to numpy array
        node_random_walk_list.append(np.array(node_random_walk_list2))  # Store processed walks
        
    return node_random_walk_list  # Return list of random walks as features



def graph_subsampling_random_node_removal(adj, rate, log=True):
    """
    Performs random node removal for graph subsampling.
    
    Parameters:
    - adj: The adjacency matrix of the graph.
    - rate: The proportion of nodes to be randomly removed.
    - log: Boolean flag to enable logging of removed nodes.
    
    Returns:
    - subsampling_graph_adj: The adjacency matrix after node removal.
    """
    # Get the total number of nodes in the graph
    node_count = len(adj)
    
    # Randomly select nodes to remove based on the given rate
    remove_node_list = random.sample(range(0, node_count), int(node_count * rate))
    remove_node_list.sort()  # Sort the list for consistency
    
    if log:
        print('remove node list:', remove_node_list)  # Log removed nodes if enabled
    
    # Step 1: Remove selected nodes and their connected edges
    subsampling_graph_adj = copy.deepcopy(adj)  # Create a deep copy of the adjacency matrix
    
    # Remove corresponding rows and columns in the adjacency matrix
    subsampling_graph_adj = np.delete(subsampling_graph_adj, remove_node_list, axis=1)
    subsampling_graph_adj = np.delete(subsampling_graph_adj, remove_node_list, axis=0)
    
    # Step 2: Identify and remove nodes that have no remaining connections
    subsampling_graph_node_count = len(subsampling_graph_adj)  # New node count after removal
    node_without_connected_edge_list = []  # List to store isolated nodes

    for i in range(subsampling_graph_node_count):
        # If a node has no connections (sum of row equals its self-loop value), mark for removal
        if sum(subsampling_graph_adj[i]) == subsampling_graph_adj[i][i]:
            node_without_connected_edge_list.append(i)
    
    # Remove isolated nodes from the adjacency matrix
    subsampling_graph_adj = np.delete(subsampling_graph_adj, node_without_connected_edge_list, axis=1)
    subsampling_graph_adj = np.delete(subsampling_graph_adj, node_without_connected_edge_list, axis=0)
    
    return subsampling_graph_adj  # Return the modified adjacency matrix
    

def graph_subsampling_random_edge_removal(adj, rate, log=True):
    """
    Randomly removes a fraction of edges from the adjacency matrix.
    
    Parameters:
    adj (numpy.ndarray): The adjacency matrix of the graph.
    rate (float): The proportion of edges to remove.
    log (bool, optional): Whether to print removed edges. Defaults to True.
    
    Returns:
    numpy.ndarray: The adjacency matrix after edge removal.
    """
    # Get the number of edges excluding self-loops
    node_count = len(adj)
    edge_count = 0
    for i in range(node_count):
        for a in range(node_count):
            if (i < a) and (adj[i][a] > 0):
                edge_count += 1
    
    # Select edges to be removed
    remove_edge_list = random.sample(range(0, edge_count), int(edge_count * rate))
    remove_edge_list.sort()
    
    if log:
        print('remove edge list:', remove_edge_list)
    
    # Remove edges from the adjacency matrix
    subsampling_graph_adj = copy.deepcopy(adj)
    count = 0
    
    for i in range(node_count):
        for a in range(node_count):
            if (i < a) and (subsampling_graph_adj[i][a] > 0):   
                if count in remove_edge_list:
                    subsampling_graph_adj[i][a] = 0
                    subsampling_graph_adj[a][i] = 0
                count += 1
    
    # Remove nodes that no longer have any connected edges
    subsampling_graph_node_count = len(subsampling_graph_adj)
    node_without_connected_edge_list = []

    for i in range(subsampling_graph_node_count):
        if sum(subsampling_graph_adj[i]) == subsampling_graph_adj[i][i]:
            node_without_connected_edge_list.append(i)
    
    subsampling_graph_adj = np.delete(subsampling_graph_adj, node_without_connected_edge_list, axis=1)
    subsampling_graph_adj = np.delete(subsampling_graph_adj, node_without_connected_edge_list, axis=0)
    
    return subsampling_graph_adj

def graph_dataset_subsampling(adj_list, node_features_list, label_list, max_neighbor_list, rate, repeat_count, node_removal=True, log=True):
    """
    Applies graph subsampling multiple times to a dataset of graphs.
    
    Parameters:
    adj_list (list of numpy.ndarray): List of adjacency matrices.
    node_features_list (list of numpy.ndarray): List of node feature matrices.
    label_list (list): List of graph labels.
    max_neighbor_list (list): List of maximum neighbor counts.
    rate (float): The proportion of edges or nodes to remove.
    repeat_count (int): Number of times to apply subsampling per graph.
    node_removal (bool, optional): If True, removes nodes instead of edges. Defaults to True.
    log (bool, optional): Whether to print logs. Defaults to True.
    
    Returns:
    tuple: Subsampled adjacency matrices, node features, labels, and max neighbor counts.
    """
    # Initialize lists to store the subsampled data
    subsampling_adj_list = []
    subsampling_node_features_list = []
    subsampling_label_list = []
    subsampling_max_neighbor_list = []
    
    for i in range(len(adj_list)):
        for a in range(repeat_count):
            if node_removal:
                subsampling_adj_list.append(graph_subsampling_random_node_removal(adj_list[i], rate, log))
            else:
                subsampling_adj_list.append(graph_subsampling_random_edge_removal(adj_list[i], rate, log))
            
            subsampling_node_features_list.append(node_features_list[i])
            subsampling_label_list.append(label_list[i])
            subsampling_max_neighbor_list.append(max_neighbor_list[i])

    return np.array(subsampling_adj_list), np.array(subsampling_node_features_list), np.array(subsampling_label_list), subsampling_max_neighbor_list


class DataReader():

    '''
    Class to read the txt files containing all data of the dataset
    '''
    def __init__(self,
                 data_dir,  # Folder with txt files
                 random_walk,
                 node2vec_hidden,
                 walk_length,
                 num_walk,
                 p,
                 q,
                 workers=3,
                 rnd_state=None,
                 use_cont_node_attr=False,  # Use or not additional float valued node attributes available in some datasets
                 folds=10):

        self.data_dir = data_dir
        self.rnd_state = np.random.RandomState() if rnd_state is None else rnd_state
        self.use_cont_node_attr = use_cont_node_attr
        files = os.listdir(self.data_dir)
        
        print('data path:', self.data_dir)
        
        data = {}
        
        # Read adj list
        nodes, graphs = self.read_graph_nodes_relations(list(filter(lambda f: f.find('graph_indicator') >= 0, files))[0]) 
        data['adj_list'] = self.read_graph_adj(list(filter(lambda f: f.find('_A') >= 0, files))[0], nodes, graphs)        
                
        print('complete to build adjacency matrix list')
        
        # Make node count list
        data['node_count_list'] = self.get_node_count_list(data['adj_list'])
        
        print('complete to build node count list')

        # Make edge matrix list
        data['edge_matrix_list'], data['max_edge_matrix'] = self.get_edge_matrix_list(data['adj_list'])
        
        print('complete to build edge matrix list')

        # Make node count list
        data['edge_matrix_count_list'] = self.get_edge_matrix_count_list(data['edge_matrix_list'])
        
        print('complete to build edge matrix count list')
        
        # Make degree_features and max neighbor list
        degree_features = self.get_node_features_degree(data['adj_list'])
        data['max_neighbor_list'] = self.get_max_neighbor(degree_features)
        
        print('complete to build max neighbor list')
       
        # Read features or make features
        if len(list(filter(lambda f: f.find('node_labels') >= 0, files))) != 0:
            print('node label: node label in dataset')
            data['features'] = self.read_node_features(list(filter(lambda f: f.find('node_labels') >= 0, files))[0], 
                                                     nodes, graphs, fn=lambda s: int(s.strip()))
        else:
            print('node label: degree of nodes')
            data['features'] = degree_features
            
        print('complete to build node features list')
        
        data['targets'] = np.array(self.parse_txt_file(list(filter(lambda f: f.find('graph_labels') >= 0, files))[0],
                                                       line_parse_fn=lambda s: int(float(s.strip()))))
                                                       
        print('complete to build targets list')
        
        if self.use_cont_node_attr:
            data['attr'] = self.read_node_features(list(filter(lambda f: f.find('node_attributes') >= 0, files))[0], 
                                                   nodes, graphs, fn=lambda s: np.array(list(map(float, s.strip().split(',')))))
        
        features, n_edges, degrees = [], [], []
        for sample_id, adj in enumerate(data['adj_list']):
            N = len(adj) # Number of nodes
            if data['features'] is not None:
                assert N == len(data['features'][sample_id]), (N, len(data['features'][sample_id]))
            n = np.sum(adj) # Total sum of edges
            n_edges.append( int(n / 2) ) # Undirected edges, so need to divide by 2
            if not np.allclose(adj, adj.T):
                print(sample_id, 'not symmetric')
            degrees.extend(list(np.sum(adj, 1)))
            features.append(np.array(data['features'][sample_id]))
                        
        # Create features over graphs as one-hot vectors for each node
        features_all = np.concatenate(features)
        features_min = features_all.min()
        features_dim = int(features_all.max() - features_min + 1) # Number of possible values
        
        features_onehot = []
        for i, x in enumerate(features):
            feature_onehot = np.zeros((len(x), features_dim))
            for node, value in enumerate(x):
                feature_onehot[node, value - features_min] = 1
            if self.use_cont_node_attr:
                feature_onehot = np.concatenate((feature_onehot, np.array(data['attr'][i])), axis=1)
            features_onehot.append(feature_onehot)

        if self.use_cont_node_attr:
            features_dim = features_onehot[0].shape[1]
            
        shapes = [len(adj) for adj in data['adj_list']]
        labels = data['targets'] # Graph class labels
        labels -= np.min(labels) # To start from 0
        N_nodes_max = np.max(shapes)

        classes = np.unique(labels)
        n_classes = len(classes)

        if not np.all(np.diff(classes) == 1):
            print('making labels sequential, otherwise pytorch might crash')
            labels_new = np.zeros(labels.shape, dtype=labels.dtype) - 1
            for lbl in range(n_classes):
                labels_new[labels == classes[lbl]] = lbl
            labels = labels_new
            classes = np.unique(labels)
            assert len(np.unique(labels)) == n_classes, np.unique(labels)
        
        print('-'*50)
        print('The number of graphs:', len(data['adj_list']))
        print('N nodes avg/std/min/max: \t%.2f/%.2f/%d/%d' % (np.mean(shapes), np.std(shapes), np.min(shapes), np.max(shapes)))
        print('N edges avg/std/min/max: \t%.2f/%.2f/%d/%d' % (np.mean(n_edges), np.std(n_edges), np.min(n_edges), np.max(n_edges)))
        print('Node degree avg/std/min/max: \t%.2f/%.2f/%d/%d' % (np.mean(degrees), np.std(degrees), np.min(degrees), np.max(degrees)))
        print('Node features dim: \t\t%d' % features_dim)
        print('N classes: \t\t\t%d' % n_classes)
        print('Classes: \t\t\t%s' % str(classes))
        for lbl in classes:
            print('Class %d: \t\t\t%d samples' % (lbl, np.sum(labels == lbl)))

        for u in np.unique(features_all):
            print('feature {}, count {}/{}'.format(u, np.count_nonzero(features_all == u), len(features_all)))
        
        N_graphs = len(labels)  # Number of samples (graphs) in data
        assert N_graphs == len(data['adj_list']) == len(features_onehot), 'invalid data'

        # Create test sets first
        train_ids, test_ids = self.split_ids(np.arange(N_graphs), rnd_state=self.rnd_state, folds=folds)

        # Create train sets
        splits = []
        for fold in range(folds):
            splits.append({'train': train_ids[fold],
                           'test': test_ids[fold]})

        data['features_onehot'] = features_onehot
        data['targets'] = labels
        data['splits'] = splits
        data['N_nodes_max'] = np.max(shapes)  # Max number of nodes
        data['features_dim'] = features_dim
        data['n_classes'] = n_classes

        # Make neighbor dictionary
        #data['neighbor_dic_list'] = self.get_neighbor_dic_list(data['adj_list'], data['N_nodes_max'])
        
        #print('complete to build neighbor dictionary list')
        
        # Make node randomwalk
        if random_walk:
            print('building node randomwalk list ...')
            data['random_walks'] = get_node_random_walk(data['features_onehot'], data['adj_list'], node2vec_hidden, walk_length, num_walk, p, q, workers)
            print('complete to build node randomwalk list')
        
        self.data = data

    def split_ids(self, ids_all, rnd_state=None, folds=10):
        n = len(ids_all)
        ids = ids_all[rnd_state.permutation(n)]
        stride = int(np.ceil(n / float(folds)))
        test_ids = [ids[i: i + stride] for i in range(0, n, stride)]
        assert np.all(np.unique(np.concatenate(test_ids)) == sorted(ids_all)), 'some graphs are missing in the test sets'
        assert len(test_ids) == folds, 'invalid test sets'
        train_ids = []
        for fold in range(folds):
            train_ids.append(np.array([e for e in ids if e not in test_ids[fold]]))
            assert len(train_ids[fold]) + len(test_ids[fold]) == len(np.unique(list(train_ids[fold]) + list(test_ids[fold]))) == n, 'invalid splits'

        return train_ids, test_ids

    def parse_txt_file(self, fpath, line_parse_fn=None):
        with open(os.path.join(self.data_dir, fpath), 'r') as f:
            lines = f.readlines()
        data = [line_parse_fn(s) if line_parse_fn is not None else s for s in lines]
        
        return data
    
    def read_graph_adj(self, fpath, nodes, graphs):
        edges = self.parse_txt_file(fpath, line_parse_fn=lambda s: s.split(','))
        adj_dict = {}
        for edge in edges:
            node1 = int(edge[0].strip()) - 1  # -1 because of zero-indexing in our code
            node2 = int(edge[1].strip()) - 1
            graph_id = nodes[node1]
            assert graph_id == nodes[node2], ('invalid data', graph_id, nodes[node2])
            if graph_id not in adj_dict:
                n = len(graphs[graph_id])
                adj_dict[graph_id] = np.zeros((n, n))
            ind1 = np.where(graphs[graph_id] == node1)[0]
            ind2 = np.where(graphs[graph_id] == node2)[0]
            assert len(ind1) == len(ind2) == 1, (ind1, ind2)
            adj_dict[graph_id][ind1, ind2] = 1
            
        adj_list = [adj_dict[graph_id] for graph_id in sorted(list(graphs.keys()))]
        
        return adj_list
        
    def read_graph_nodes_relations(self, fpath):
        graph_ids = self.parse_txt_file(fpath, line_parse_fn=lambda s: int(s.rstrip()))
        nodes, graphs = {}, {}
        for node_id, graph_id in enumerate(graph_ids):
            if graph_id not in graphs:
                graphs[graph_id] = []
            graphs[graph_id].append(node_id)
            nodes[node_id] = graph_id
        graph_ids = np.unique(list(graphs.keys()))
        for graph_id in graphs:
            graphs[graph_id] = np.array(graphs[graph_id])
        return nodes, graphs

    def read_node_features(self, fpath, nodes, graphs, fn):
        node_features_all = self.parse_txt_file(fpath, line_parse_fn=fn)
        node_features = {}
        
        for node_id, x in enumerate(node_features_all):
            graph_id = nodes[node_id]
            if graph_id not in node_features:
                node_features[graph_id] = [ None ] * len(graphs[graph_id])
            ind = np.where(graphs[graph_id] == node_id)[0]
            assert len(ind) == 1, ind
            assert node_features[graph_id][ind[0]] is None, node_features[graph_id][ind[0]]
            node_features[graph_id][ind[0]] = x
        node_features_lst = [node_features[graph_id] for graph_id in sorted(list(graphs.keys()))]
        return node_features_lst

    def get_node_features_degree(self, adj_list):
        node_features_list = []

        for adj in adj_list:
            sub_list = []
            for feature in nx.from_numpy_array(np.array(adj)).degree():
                sub_list.append(feature[1])
            node_features_list.append(np.array(sub_list))

        return node_features_list
        
    def get_max_neighbor(self, degree_list):
        max_neighbor_list = []
        
        for degrees in degree_list:
            max_neighbor_list.append(int(max(degrees)))

        return max_neighbor_list

    def get_node_count_list(self, adj_list):
        node_count_list = []
        
        for adj in adj_list:
            node_count_list.append(len(adj))
                        
        return node_count_list

    def get_edge_matrix_list(self, adj_list):
        edge_matrix_list = []
        max_edge_matrix = 0
        
        for adj in adj_list:
            edge_matrix = []
            for i in range(len(adj)):
                for j in range(len(adj[0])):
                    if adj[i][j] == 1:
                        edge_matrix.append((i,j))
            if len(edge_matrix) > max_edge_matrix:
                max_edge_matrix = len(edge_matrix)
            edge_matrix_list.append(np.array(edge_matrix))
                        
        return edge_matrix_list, max_edge_matrix

    def get_edge_matrix_count_list(self, edge_matrix_list):
        edge_matrix_count_list = []
        
        for edge_matrix in edge_matrix_list:
            edge_matrix_count_list.append(len(edge_matrix))
                        
        return edge_matrix_count_list
    
    def get_neighbor_dic_list(self, adj_list, N_nodes_max):
        neighbor_dic_list = []
        
        for adj in adj_list:
            neighbors = []
            for i, row in enumerate(adj):
                idx = np.where(row == 1.0)[0].tolist()
                idx = np.pad(idx, (0, N_nodes_max - len(idx)), 'constant', constant_values=0)
                neighbors.append(idx)
            for a in range(i, N_nodes_max - 1):
                neighbors.append(np.array([0]*136))
            neighbor_dic_list.append(np.array(neighbors))
        
        return neighbor_dic_list    

class GraphData(torch.utils.data.Dataset):
    def __init__(self,
                 fold_id,
                 datareader,
                 split,
                 random_walk,
                 n_graph_subsampling,
                 graph_node_subsampling,
                 graph_subsampling_rate):
        
        self.random_walk = random_walk
        
        self.set_fold(datareader.data, split, fold_id, n_graph_subsampling, graph_node_subsampling, graph_subsampling_rate)

    def set_fold(self, data, split, fold_id, n_graph_subsampling, graph_node_subsampling, graph_subsampling_rate):
        self.total = len(data['targets'])
        self.N_nodes_max = data['N_nodes_max']
        self.max_edge_matrix = data['max_edge_matrix']
        self.n_classes = data['n_classes']
        self.features_dim = data['features_dim']
        self.idx = data['splits'][fold_id][split]
        
        # Use deepcopy to make sure we don't alter objects in folds
        self.labels = copy.deepcopy([data['targets'][i] for i in self.idx])
        self.adj_list = copy.deepcopy([data['adj_list'][i] for i in self.idx])
        self.features_onehot = copy.deepcopy([data['features_onehot'][i] for i in self.idx])
        self.max_neighbor_list = copy.deepcopy([data['max_neighbor_list'][i] for i in self.idx])
        self.edge_matrix_list = copy.deepcopy([data['edge_matrix_list'][i] for i in self.idx])
        self.node_count_list = copy.deepcopy([data['node_count_list'][i] for i in self.idx])
        self.edge_matrix_count_list = copy.deepcopy([data['edge_matrix_count_list'][i] for i in self.idx])
        #self.neighbor_dic_list = copy.deepcopy([data['neighbor_dic_list'][i] for i in self.idx])
        
        if self.random_walk:
            self.random_walks = copy.deepcopy([data['random_walks'][i] for i in self.idx])
        
        if n_graph_subsampling:
            self.adj_list, self.features_onehot, self.labels, self.max_neighbor_list, self.neighbor_dic_list = graph_dataset_subsampling(self.adj_list,
                                                                                   self.features_onehot, 
                                                                                   self.labels,
                                                                                   self.max_neighbor_list,
                                                                                   rate=graph_subsampling_rate,
                                                                                   repeat_count=n_graph_subsampling,
                                                                                   node_removal=graph_node_subsampling,
                                                                                   log=False)
        
        print('%s: %d/%d' % (split.upper(), len(self.labels), len(data['targets'])))
        
        # Sample indices for this epoch
        if n_graph_subsampling:
            self.indices = np.arange(len(self.idx) * n_graph_subsampling)  
        else:
            self.indices = np.arange(len(self.idx))
        
    def pad(self, mtx, desired_dim1, desired_dim2=None, value=0, mode='edge_matrix'):
        sz = mtx.shape
        #assert len(sz) == 2, ('only 2d arrays are supported', sz)
        
        if len(sz) == 2:
            if desired_dim2 is not None:
                  mtx = np.pad(mtx, ((0, desired_dim1 - sz[0]), (0, desired_dim2 - sz[1])), 'constant', constant_values=value)
            else:
                  mtx = np.pad(mtx, ((0, desired_dim1 - sz[0]), (0, 0)), 'constant', constant_values=value)
        elif len(sz) == 3:
            mtx = np.pad(mtx, ((0, desired_dim1 - sz[0]), (0, 0), (0, 0)), 'constant', constant_values=value)
        
        return mtx
    
    def nested_list_to_torch(self, data):
        #if isinstance(data, dict):
            #keys = list(data.keys())           
        for i in range(len(data)):
            #if isinstance(data, dict):
                #i = keys[i]
            if isinstance(data[i], np.ndarray):
                data[i] = torch.from_numpy(data[i]).float()
            #elif isinstance(data[i], list):
                #data[i] = list_to_torch(data[i])
        return data
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        index = self.indices[index]
        N_nodes_max = self.N_nodes_max
        N_nodes = self.adj_list[index].shape[0]
        graph_support = np.zeros(self.N_nodes_max)
        graph_support[:N_nodes] = 1
        
        if self.random_walk:
            return self.nested_list_to_torch([self.pad(self.features_onehot[index].copy(), self.N_nodes_max),  # Node_features
                                          self.pad(self.adj_list[index], self.N_nodes_max, self.N_nodes_max),  # Adjacency matrix
                                          graph_support,  # Mask with values of 0 for dummy (zero padded) nodes, otherwise 1 
                                          N_nodes,
                                          int(self.labels[index]),
                                          int(self.max_neighbor_list[index]),
                                          self.pad(self.random_walks[index])])                           
        else:
            return self.nested_list_to_torch([self.pad(self.features_onehot[index].copy(), self.N_nodes_max),  # Node_features
                                          self.pad(self.adj_list[index], self.N_nodes_max, self.N_nodes_max),  # Adjacency matrix
                                          graph_support,  # Mask with values of 0 for dummy (zero padded) nodes, otherwise 1 
                                          N_nodes,
                                          int(self.labels[index]),
                                          int(self.max_neighbor_list[index]),
                                          self.pad(self.edge_matrix_list[index], self.max_edge_matrix),
                                          int(self.node_count_list[index]),
                                          int(self.edge_matrix_count_list[index])])            
