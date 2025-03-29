import numpy as np
import itertools
from datasets.dataset import *



class grid: 
    
    def __init__(self, params_dict): 
        self.params_dict = params_dict

    def get_combinations(self):
        keys, values = zip(*self.param_dict.items())
        return [dict(zip(keys, v)) for v in itertools.product(*values)]

    def __repr__(self):
        return f"HyperparameterGrid({self.param_dict})"
    

class ModelSelection(): 
    
    def __init__(self, DATASET, data_split, model_class, grid): 
        self.model_class = model_class
        self.grid = grid
        self.DATASET = DATASET
        self.train_split = GraphDatasetSubset(self.DATASET.dataset.get_data(), data_split["train"])
        self.validation_split = GraphDatasetSubset(self.DATASET.dataset.get_data(), data_split["validation"])
        self.best_config = None

    def fit(self):
        for config in self.grid.get_combinations():
            print(f"Training with configuration: {config}")
            
            model = self.model_class(**config)
            model.fit(self.train_split)
            accuracy = model.evaluate(self.validation_split)

            print(f"Validation accuracy: {accuracy}")

            if self.best_config is None or accuracy > self.best_config["accuracy"]:
                self.best_config = {"config": config, "accuracy": accuracy}
        
    def get_best_config(self):
        if self.best_config is None:
            raise ValueError("No configurations have been evaluated.")
        return self.best_config["config"], self.best_config["accuracy"]
    