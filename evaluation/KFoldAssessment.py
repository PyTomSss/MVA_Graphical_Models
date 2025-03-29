import numpy as np
from tqdm import tqdm
from datasets.dataset import GraphDatasetSubset
from evaluation.ModelSelection import *

class Holdout(): 
    def __init__(self, DATASET, train_split, train_size=0.9): 
        self.DATASET = DATASET
        self.train_split = train_split
        self.train_size = train_size
        self.num_samples = len(self.train_split)
        self.train_indices = None

    def split(self):

        indices = np.arange(self.num_samples)
        np.random.shuffle(indices)
        self.train_indices = indices[:int(self.num_samples * self.train_size)]
       

    def get_splits(self):
        if self.train_indices is None :
            raise ValueError("Data has not been split yet.")
        return self.train_indices
    


class ModelAssessment(): 
    
    def __init__(self, model_class, DATASET, grid, Kfold=10, holdout=3): 
        self.model_class = model_class
        self.DATASET = DATASET
        self.grid = grid
        self.Kfold = Kfold
        self.best_configs = []
        self.splits = DATASET.splits
        self.holdout = holdout
        self.fold_accuracy = []

    def assess(self):

        for i in tqdm(range(self.Kfold)):
            train_data_idxs = self.splits[i]["model_selection"][0]
            test_data_idxs = self.splits[i]["test"]
            test_data = GraphDatasetSubset(self.DATASET.dataset.get_data(), test_data_idxs)

            model_selection = ModelSelection(self.DATASET, train_data_idxs, self.model_class, self.grid)
            model_selection.fit()
            best_config, _ = model_selection.get_best_config()
            self.best_configs.append(best_config)
            inner_accuracy = []

            for j in range(self.holdout): 
                
                train_holdout = Holdout(self.DATASET, train_data_idxs)
                train_holdout.split()
                train_indices = train_holdout.get_splits()
                
                train_data = GraphDatasetSubset(self.DATASET.dataset.get_data(), train_indices)
                model = self.model_class(**best_config)
                model.fit(train_data)
                accuracy = model.evaluate(test_data)
                inner_accuracy.append(accuracy)
            
            self.fold_accuracy.append(np.mean(inner_accuracy))
        
        return np.mean(self.fold_accuracy), np.std(self.fold_accuracy)