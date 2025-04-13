import itertools
import random
from datasets.dataset import GraphDatasetSubset
from Pipeline.train import Train

# ------------------------------------------------------------------------
# Class: grid
# Description: Generates all combinations of hyperparameters from a grid.
# ------------------------------------------------------------------------

class grid: 
    
    def __init__(self, params_dict): 
        self.params_dict = params_dict

    def get_combinations(self):
        keys, values = zip(*self.params_dict.items())
        return [dict(zip(keys, v)) for v in itertools.product(*values)]

    def __repr__(self):
        return f"HyperparameterGrid({self.params_dict})"
    

# ------------------------------------------------------------------------
# Class: ModelSelection
# Description: Performs model selection by testing different hyperparameter
# configurations (via Grid Search or Random Search) on a validation set.
# ------------------------------------------------------------------------

class ModelSelection(): 
    
    def __init__(self, DATASET, data_split, grid, random_search=False, n_samples=10): 

        self.grid = grid
        self.DATASET = DATASET
        self.train_split = GraphDatasetSubset(self.DATASET.dataset.get_data(), data_split["train"])
        self.validation_split = GraphDatasetSubset(self.DATASET.dataset.get_data(), data_split["validation"])
        self.best_config = None
        self.random_search = random_search
        self.n_samples = n_samples

    def fit(self):
        dic = {"lr" : self.grid["lr"], "num_lay" : self.grid["num_lay"], "hidden_agg_lay_size" : self.grid["hidden_agg_lay_size"]}
        dic = {"lr" : self.grid["lr"]}
        grille = grid(dic)
        all_combinations = grille.get_combinations()
        
        if self.random_search:
            combinations_to_try = random.sample(all_combinations, min(self.n_samples, len(all_combinations)))
        else:
            combinations_to_try = all_combinations

        for config in combinations_to_try:
            print(f"[Model Selection] - Training with configuration: {config} \n")
            params = self.grid
            params["lr"] = config["lr"]
            
            params["num_lay"] = config["num_lay"]
            params["hidden_agg_lay_size"] = config["hidden_agg_lay_size"]


            trainer = Train(params, data = self.train_split)
            trainer.fit()
            accuracy = trainer.evaluate(self.validation_split, graphDATA = True)

            if self.best_config is None or accuracy > self.best_config["accuracy"]:
                self.best_config = {"config": params, "accuracy": accuracy}
        
    def get_best_config(self):
        if self.best_config is None:
            raise ValueError("No configurations have been evaluated.")
        return self.best_config["config"], self.best_config["accuracy"]
