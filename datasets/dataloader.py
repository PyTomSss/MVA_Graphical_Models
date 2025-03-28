import torch.utils.data

from .data import GraphBatch


class DataLoader(torch.utils.data.DataLoader):
    r"""Data loader qui merge les données en mini-batchs. 
    Prend en argument: 
    - dataset : le dataset à charger
    - batch_size : la taille du batch
    - shuffle : si True, les données sont mélangées à chaque époque
    - follow_batch : liste des clés pour lesquelles on veut créer un vecteur de batch
    """

    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 follow_batch=[],
                 **kwargs):
        super(DataLoader, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: Batch.from_data_list(
                data_list, follow_batch),
            **kwargs)