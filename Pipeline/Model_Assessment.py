import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from Model_Selection import ModelSelection
from Pipeline.Train import Train
from datasets.dataset import GraphDatasetSubset


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
    
    def __init__(self, DATASET, grid, Kfold=10, holdout=3): 
        self.DATASET = DATASET
        self.grid = grid
        self.Kfold = Kfold
        self.best_configs = None
        self.splits = DATASET.splits
        self.holdout = holdout
        self.fold_accuracy = []

    def assess(self):

        for i in tqdm(range(self.Kfold)):
            train_data_idxs = self.splits[i]["model_selection"][0]
            test_data_idxs = self.splits[i]["test"]
            test_data = GraphDatasetSubset(self.DATASET.dataset.get_data(), test_data_idxs)

            model_selection = ModelSelection(self.DATASET, train_data_idxs, self.grid.copy())
            model_selection.fit()
            best_config, _ = model_selection.get_best_config()
            self.best_configs = best_config
            inner_accuracy = []

            for j in range(self.holdout): 
                
                train_holdout = Holdout(self.DATASET, train_data_idxs)
                train_holdout.split()
                train_indices = train_holdout.get_splits()
                
                train_data = GraphDatasetSubset(self.DATASET.dataset.get_data(), train_indices)
                model = Train(self.best_configs, train_data)
                model.fit()
                accuracy = model.evaluate(test_data, graphDATA = True)
                inner_accuracy.append(accuracy)
            
            self.fold_accuracy.append(np.mean(inner_accuracy))
        
        return np.mean(self.fold_accuracy), np.std(self.fold_accuracy)
    



def plot_gnn_comparison(models, datasets, means_1, stds_1, means_2, stds_2, title="GNN Comparison", label_1="Random Search", label_2="Grid Search"):
    """
    Trace un graphique comparant la performance de modèles GNN avec Random Search et Grid Search.

    Parameters:
    - models : list[str] — noms des modèles (['GCN', 'GAT', 'GIN', 'SAGE', 'Baseline'])
    - datasets : list[str] — noms des datasets (['IMDB-Binary', 'PROTEINS', 'D&D'])
    - means_1 : dict[dataset][model] — moyenne des accuracies (méthode 1)
    - stds_1 : dict[dataset][model] — écart-type des accuracies (méthode 1)
    - means_2 : dict[dataset][model] — moyenne des accuracies (méthode 2)
    - stds_2 : dict[dataset][model] — écart-type des accuracies (méthode 2)
    """

    colors = {
        'GCN': 'blue',
        'GAT': 'orange',
        'GIN': 'green',
        'SAGE': 'red',
        'Baseline': 'purple'
    }
    
    n_datasets = len(datasets)
    fig, axes = plt.subplots(1, n_datasets, figsize=(16, 5), sharey=True)
    fig.suptitle(title, fontsize=14)

    for i, dataset in enumerate(datasets):
        ax = axes[i]
        x = np.arange(len(models))
        offset = 0.15

        for j, model in enumerate(models):
            color = colors.get(model, 'black')
            
            ax.errorbar(j - offset, means_1[dataset][model], 
                        yerr=stds_1[dataset][model],
                        fmt='o', color=color, label=label_1 if j == 0 else "",
                        capsize=4)
        
            ax.errorbar(j + offset, means_2[dataset][model], 
                        yerr=stds_2[dataset][model],
                        fmt='s', color=color, label=label_2 if j == 0 else "",
                        capsize=4)

        ax.set_title(dataset)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True)

    axes[0].legend(loc='lower left')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
