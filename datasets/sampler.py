import torch
from torch.utils.data import sampler


class RandomSampler(sampler.RandomSampler):
    """
    Permet de sauvegarder une permutation des données d'entrainements dans l'attribut 'permutation'.
    
    Voici donc comment instancier un DataLoader avec un RandomSampler:

    >>> data = Dataset()
    >>> dataloader = DataLoader(dataset=data, batch_size=32, shuffle=False, sampler=RandomSampler(data))
    >>> for batch in dataloader:
    >>>     print(batch)
    >>> print(dataloader.sampler.permutation)
    """

    def __init__(self, data_source, num_samples=None, replacement=False):
        super().__init__(data_source, replacement=replacement, num_samples=num_samples)
        self.permutation = None

    def __iter__(self):
        n = len(self.data_source)
        self.permutation = torch.randperm(n).tolist()
        return iter(self.permutation)