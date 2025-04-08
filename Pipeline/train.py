# ==============================================================================
# File: train.py
#
# Description:
# In this file, we define a Trainer class that handles the training and evaluation of Graph Neural Networks (GNNs).
# The class facilitates:
# - Model instantiation based on the dataset and model type specified in params.
# - Data loading, optimization, and learning rate scheduling.
# - Training and evaluation of the model for multiple epochs.
# - Early stopping for optimal training duration based on test accuracy.
# ==============================================================================

from datasets.manager import IMDBBinary, DD
from tqdm import tqdm
import torch 
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from src.models import GCN, GAT, GraphSAGE, GIN
import time
from utils.utils import get_adjacency_and_features
from datasets.dataset import GraphDatasetSubset


class Train:
    def __init__(self, params, data = None):
        """
        Initialize the Trainer class with the provided parameters.

        Args:
            params (dict): Dictionary containing training and model parameters. The structure of the dictionary is as follows:
                - 'model_type' (str): The type of model to train ("GCN", "GAT", "GIN", "GraphSAGE").
                - 'n_graph_subsampling' (int): The number of graph subsampling operations to apply during training.
                - 'graph_node_subsampling' (bool): Whether to perform node subsampling (True) or edge subsampling (False).
                - 'graph_subsampling_rate' (float): The rate at which to subsample the graph.
                - 'dataset' (str): The dataset to use for training ("IMDB" or "DD").
                - 'pooling_type' (str): Type of pooling to apply ("mean" or others).
                - 'seed' (int): Random seed for reproducibility.
                - 'n_folds' (int): Number of cross-validation folds.
                - 'cuda' (bool): Whether to use GPU for training (True or False).
                - 'lr' (float): Learning rate for the optimizer.
                - 'epochs' (int): Number of epochs to train the model.
                - 'weight_decay' (float): Weight decay for regularization.
                - 'batch_size' (int): Batch size for training.
                - 'dropout' (float): Dropout rate for the model layers.
                - 'num_lay' (int): Number of layers in the model.
                - 'num_agg_layer' (int): Number of aggregation layers in the GNN.
                - 'hidden_agg_lay_size' (int): Size of the hidden layers for graph aggregation.
                - 'fc_hidden_size' (int): Size of fully-connected layers after readout.
                - 'threads' (int): Number of subprocesses for data loading.
                - 'random_walk' (bool): Whether to perform random walk-based graph augmentation.
                - 'walk_length' (int): Length of each random walk.
                - 'num_walk' (int): Number of random walks.
                - 'p' (float): Probability to return to the previous vertex during random walks.
                - 'q' (float): Probability of moving away from the previous vertex during random walks.
                - 'print_logger' (int): Frequency (in epochs) of printing the training logs.
                - 'eps' (float): Epsilon value for the GIN model (only used in GIN).
                
        Note:
            - The dataset is automatically loaded depending on the 'dataset' specified in params.
            - Only supports 'IMDB' or 'DD' datasets as currently coded.
        """
        if params["dataset"] == "IMDB":
            self.dataset = IMDBBinary()
        elif params["dataset"] == "DD":
            self.dataset = DD()

        self.params = params

        # Extract the graph data and their corresponding labels
        if data: 
            self.x_dataset, self.y_dataset = data, data.get_targets()
            print(f"taille de x_dataset: {len(self.x_dataset)} et taille de y_dataset : {len(self.y_dataset)}")
        else: 
            self.x_dataset, self.y_dataset = self.dataset.dataset.get_data(), self.dataset.dataset.get_targets()

        # Select device (GPU if available and requested, else CPU)
        if self.params["cuda"] and torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        self.model = self.get_model()  # Instantiate the model
        self.loss_fn = F.cross_entropy  # Loss function for classification tasks

    def get_model(self):
        """
        Instantiates and returns the model specified in the params dictionary.

        Args:
            None

        Returns:
            torch.nn.Module: The GNN model instantiated based on the model type in params["model_type"].
        """
        input_features = self.x_dataset[0].x.shape[1]  # Number of input node features

        # Model selection based on the 'model_type' parameter
        if self.params["model_type"] == 'GCN':
            model = GCN(
                n_feat=input_features,
                n_class=2,
                n_layer=self.params['num_agg_layer'],
                agg_hidden=self.params['hidden_agg_lay_size'],
                fc_hidden=self.params['fc_hidden_size'],
                dropout=self.params['dropout'],
                pool_type=self.params['pooling_type'],
                device=self.device
            ).to(self.device)

        elif self.params["model_type"] == 'GAT':
            model = GAT(
                n_feat=input_features,
                n_class=2,
                n_layer=self.params['num_agg_layer'],
                agg_hidden=self.params['hidden_agg_lay_size'],
                fc_hidden=self.params['fc_hidden_size'],
                dropout=self.params['dropout'],
                pool_type=self.params['pooling_type'],
                device=self.device
            ).to(self.device)

        elif self.params["model_type"] == 'GraphSAGE':
            model = GraphSAGE(
                n_feat=input_features,
                n_class=2,
                n_layer=self.params['num_agg_layer'],
                agg_hidden=self.params['hidden_agg_lay_size'],
                fc_hidden=self.params['fc_hidden_size'],
                dropout=self.params['dropout'],
                pool_type=self.params["pooling_type"],
                device=self.device
            ).to(self.device)

        elif self.params["model_type"] == 'GIN':
            model = GIN(
                n_feat=input_features,
                n_class=2,
                n_layer=self.params['num_agg_layer'],
                agg_hidden=self.params['hidden_agg_lay_size'],
                fc_hidden=self.params['fc_hidden_size'],
                dropout=self.params['dropout'],
                pool_type=self.params["pooling_type"],
                device=self.device
            ).to(self.device)

        return model

    def loaders_train_test_setup(self):
        """
        Sets up the DataLoader, optimizer, and learning rate scheduler for training.

        Args:
            None

        Returns:
            Tuple: (DataLoader, optimizer, scheduler)
                - DataLoader (torch.utils.data.DataLoader): DataLoader for training.
                - optimizer (torch.optim.Optimizer): Optimizer for training the model.
                - scheduler (torch.optim.lr_scheduler): Learning rate scheduler for adjusting the learning rate during training.
        """
        # Create a custom DataLoader that simply returns indices (no batching)
        loader = torch.utils.data.DataLoader(
            range(len(self.x_dataset)),
            batch_size=1,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
            collate_fn=lambda x: x  # x will be a list of one index
        )

        # Count and display number of trainable parameters
        c = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('N trainable parameters:', c)

        # Define Adam optimizer with weight decay
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.params["lr"],
            weight_decay=self.params["weight_decay"],
            betas=(0.5, 0.999)
        )

        # Define learning rate scheduler (reduce LR at epochs 20 and 30)
        scheduler = lr_scheduler.MultiStepLR(optimizer, [20, 30], gamma=0.1)

        return loader, optimizer, scheduler

    def train(self, train_loader, optimizer, scheduler, epoch):
        """
        Perform one training epoch over the dataset.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
            optimizer (torch.optim.Optimizer): Optimizer for training the model.
            scheduler (torch.optim.lr_scheduler): Learning rate scheduler to adjust the learning rate.
            epoch (int): The current epoch index.

        Returns:
            float: Average time per iteration during this epoch.
        """
        self.model.train()
        train_loss, n_samples = 0, 0
        total_time_iter = 0
        start = time.time()

        for batch_idx, data_batch in enumerate(train_loader):
            
            idx = data_batch[0]  # Extract index

            x = self.x_dataset[idx]
            y = self.y_dataset[idx]

            optimizer.zero_grad()

            if self.params["model_type"] == "GraphSAGE":
                output = self.model(x)  # Forward pass for GraphSAGE
            else:
                A, f = get_adjacency_and_features(x)
                A = A.to(self.device)
                f = f.to(self.device)
                y = torch.tensor([y], device=self.device)

                output = self.model(f, A)  # Forward pass for other models

            y = torch.tensor([y], device=self.device)  # Wrap in tensor for batch dimension

            loss = self.loss_fn(output.unsqueeze(0), y)  # Add batch dimension to output

            loss.backward()
            optimizer.step()

            # Timing and logging
            time_iter = time.time() - start
            total_time_iter += time_iter
            train_loss += loss.item()
            n_samples += 1

            #if batch_idx % self.params["print_logger"] == 0 or batch_idx == len(train_loader) - 1:
                #print(f'Train Epoch: {epoch} [{n_samples}/{len(train_loader.dataset)} ({100. * (batch_idx + 1) / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f} (avg: {train_loss / n_samples:.6f}) \tsec/iter: {time_iter / (batch_idx + 1):.4f}')

            start = time.time()  # Reset timer

        scheduler.step()  # Adjust learning rate
        return total_time_iter / (len(train_loader) + 1)

    def evaluate(self, test_loader, graphDATA = False):
        """
        Evaluate the model on the test set.

        Args:
            test_loader (torch.utils.data.DataLoader): DataLoader for the test data.

        Returns:
            float: Accuracy on the test set.
        """
        self.model.eval()
        correct, n_samples = 0, 0

        if graphDATA: 
            _loader = torch.utils.data.DataLoader(
            range(len(test_loader)),
            batch_size=1,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
            collate_fn=lambda x: x  # x will be a list of one index
            ) 

            targets = test_loader.get_targets()
            
            with torch.no_grad():
                for batch_idx, data_batch in enumerate(_loader):
                    idx = data_batch[0]
                    x = test_loader[idx]
                    y = targets[idx]

                    if self.params["model_type"] == "GraphSAGE":
                        output = self.model(x)
                    else:
                        A, f = get_adjacency_and_features(x)
                        A = A.to(self.device)
                        f = f.to(self.device)
                        y = torch.tensor([y], device=self.device)

                        output = self.model(f, A)

                    # Prediction: binary or multi-class
                    if output.shape[-1] == 1:
                        pred = (torch.sigmoid(output) > 0.5).long()
                    else:
                        pred = output.argmax(dim=-1)

                    correct += (pred == y).sum().item()
                    n_samples += 1

        else: 
            with torch.no_grad():
                for batch_idx, data_batch in enumerate(test_loader):
                    idx = data_batch[0]
                    x = self.x_dataset[idx]
                    y = self.y_dataset[idx]

                    if self.params["model_type"] == "GraphSAGE":
                        output = self.model(x)
                    else:
                        A, f = get_adjacency_and_features(x)
                        A = A.to(self.device)
                        f = f.to(self.device)
                        y = torch.tensor([y], device=self.device)

                        output = self.model(f, A)

                    # Prediction: binary or multi-class
                    if output.shape[-1] == 1:
                        pred = (torch.sigmoid(output) > 0.5).long()
                    else:
                        pred = output.argmax(dim=-1)

                    correct += (pred == y).sum().item()
                    n_samples += 1

        acc = 100. * correct / n_samples
        #print(f'Test set (epoch {self.params["epochs"]}): Accuracy: {correct}/{n_samples} ({acc:.2f}%)\n')
        print(f'Accuracy: {correct}/{n_samples} ({acc:.2f}%)\n')

        return acc

    def fit(self):
        """
        Run the full training and evaluation loop.

        Returns:
            list: [dataset name, dataset name (again), best accuracy achieved]
        """
        loader, optimizer, scheduler = self.loaders_train_test_setup()
        total_time = 0
        best_acc = 0
        patience_counter = 0
        patience = self.params.get("early_stopping_patience", 5)

        for epoch in tqdm(range(self.params["epochs"]), desc="Epochs", position=1, leave=False):
            total_time_iter = self.train(loader, optimizer, scheduler, epoch)
            total_time += total_time_iter
            acc = self.evaluate(loader)  # Same loader used for train/test (no split)

            # Early stopping logic
            if acc > best_acc:
                best_acc = acc
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

        print(f'Best Accuracy: {best_acc:.2f}%')
        print(f'Average training time per epoch AND PER DATA???: {total_time / (epoch + 1):.2f} seconds')

        return [self.params["dataset"], self.params["dataset"], best_acc]
