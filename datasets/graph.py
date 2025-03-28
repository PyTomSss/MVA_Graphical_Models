import networkx as nx
import torch


class Graph(nx.Graph):
    def __init__(self, target, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target = target
        self.laplacians = None
        self.v_plus = None

    def get_edge_index(self):
        """remplace la méthode de PyG "dense_to_sparse" en utilisant directement les arêtes du graphe"""
        edge_index = torch.tensor(list(self.edges), dtype=torch.long).t().contiguous()
        return edge_index

    def get_edge_attr(self):
        features = []
        for _, _, edge_attrs in self.edges(data=True):
            data = []

            if "label" in edge_attrs and edge_attrs["label"] is not None:
                data.extend(edge_attrs["label"])

            if "attrs" in edge_attrs and edge_attrs["attrs"] is not None:
                data.extend(edge_attrs["attrs"])

            features.append(data if data else [0])  # Si aucune feature, ajoute une valeur par défaut
        return torch.tensor(features, dtype=torch.float)

    def get_x(self, use_node_attrs=False, use_node_degree=False, use_one=False):
        
        features = []
        for node, node_attrs in self.nodes(data=True):
            data = []

            if "label" in node_attrs and node_attrs["label"] is not None:
                data.extend(node_attrs["label"])

            if use_node_attrs and "attrs" in node_attrs and node_attrs["attrs"] is not None:
                data.extend(node_attrs["attrs"])

            if use_node_degree:
                data.append(self.degree(node))

            if use_one:
                data.append(1)
            
            features.append(data if data else [0])  # Valeur par défaut si aucun attribut
        return torch.tensor(features, dtype=torch.float)

    def get_target(self, classification=True):
        
        return torch.tensor([self.target], dtype=torch.long if classification else torch.float)

    @property
    def has_edge_attrs(self):
        return any("attrs" in edge_attrs and edge_attrs["attrs"] is not None for _, _, edge_attrs in self.edges(data=True))

    @property
    def has_edge_labels(self):
        return any("label" in edge_attrs and edge_attrs["label"] is not None for _, _, edge_attrs in self.edges(data=True))

    @property
    def has_node_attrs(self):
        return any("attrs" in node_attrs and node_attrs["attrs"] is not None for _, node_attrs in self.nodes(data=True))

    @property
    def has_node_labels(self):
        return any("label" in node_attrs and node_attrs["label"] is not None for _, node_attrs in self.nodes(data=True))
