import torch

class GraphData:
    def __init__(self,
                 x=None, 
                 edge_index=None, 
                 edge_attr=None, 
                 y=None,
                 v_outs=None, 
                 e_outs=None, 
                 g_outs=None, 
                 o_outs=None,
                 laplacians=None, 
                 v_plus=None):

        self.x = x  # node features
        self.edge_index = edge_index  # indices des arêtes
        self.edge_attr = edge_attr  # edges features
        self.y = y  # label du graph (graph classification)

        self.v_outs = v_outs # embedding des noeuds
        self.e_outs = e_outs # embedding des arêtes
        self.g_outs = g_outs # embedding du graph (dans sa globalité)
        self.o_outs = o_outs # autres outputs
        self.laplacians = laplacians # laplaciens pour les convolutions spectrales
        self.v_plus = v_plus

    def __repr__(self):
        return f"GraphData(x={self.x.shape if self.x is not None else None}, " \
               f"edge_index={self.edge_index.shape if self.edge_index is not None else None})"


class GraphBatch:
    @staticmethod
    def from_data_list(data_list):
        """Fonction permettant de concaténer plusieurs graphes en un batch"""
        
        laplacians = None
        v_plus = None

        if hasattr(data_list[0], 'laplacians') and data_list[0].laplacians is not None:
            laplacians = [d.laplacians for d in data_list]
            v_plus = [d.v_plus for d in data_list]

        x = torch.cat([d.x for d in data_list], dim=0) if data_list[0].x is not None else None
        edge_attr = torch.cat([d.edge_attr for d in data_list], dim=0) if data_list[0].edge_attr is not None else None
        y = torch.cat([d.y for d in data_list], dim=0) if data_list[0].y is not None else None

        # on ajuste les indices pour correspondre au batch
        edge_index_list = []
        node_offset = 0

        for d in data_list:
            edge_index_list.append(d.edge_index + node_offset)
            node_offset += d.x.shape[0] if d.x is not None else 0  

        edge_index = torch.cat(edge_index_list, dim=1)

        return GraphData(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                         v_outs=None, e_outs=None, g_outs=None, o_outs=None,
                         laplacians=laplacians, v_plus=v_plus)

