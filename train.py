import argparse
from argparse import RawTextHelpFormatter
import sys
import numpy as np
import time
import statistics

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from src.models import GCN, GAT, GraphDenseNet

from src.graphdata import DataReader, GraphData


'''
Possible values: 

model_list = ['GCN', 'GAT', 'GraphDenseNet']
dataset_list = ['IMDB-BINARY', 'DD']
readout_list = ['max', 'avg', 'sum']

'''

parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

# DEFAULT PARAMS 

params_list = {"model_type": "GCN",  
               "n_graph_subsampling": 0, # the number of running graph subsampling each train graph data run subsampling 5 times: increasing graph data 5 times
               "graph_node_subsampling": True, # TRUE: removing node randomly to subsampling and augmentation of graph dataset \n'+
                # FALSE: removing edge randomly to subsampling and augmentation of graph dataset
               "graph_subsampling_rate": 0.2, # graph subsampling rate
               "dataset": "IMDB", 
               "pooling_type": "mean", 
               "seed": 42,
               "n_folds": 10, 
               "cuda": True, 
               "lr": 0.001, 
               "epochs": 50, 
               "weight_decay":5e-4,
               "batch_size": 32, 
               "dropout": 0, # dropout rate of layer
               "num_lay": 5, 
               "num_agg_layer": 2, # the number of graph aggregation layers
               "hidden_agg_lay_size": 64, # size of hidden graph aggregation layer
               "fc_hidden_size": 128, # size of fully-connected layer after readout
               "threads":10, # how many subprocesses to use for data loading
               "random_walk":True,
               "walk_length": 20, # walk length of random walk, 
               "num_walk": 10, # num of random walk
               "p": 0.65, # Possibility to return to the previous vertex, how well you navigate around
               "q": 0.35, # Possibility of moving away from the previous vertex, how well you are exploring new places
               "print_logger": 10  # printing rate
               }


class Training:
    def __init__(self, params):
      self.params = params 

      if self.params["cuda"] and torch.cuda.is_available():
          self.device = "cuda:0"
      else: 
          self.device = "cpu"

      self.datareader = self.get_dataset()
      self.model = self.get_model()

      self.acc_folds, self.time_folds = [], []

      self.loss_fn = F.cross_entropy 

    
    def get_dataset(self):
          
      # Build graph data reader: IMDB-BINARY, IMDB-MULTI, ...
      if self.params["dataset"] == "IMDB":
          self.data_dir = 'data/IMDB-BINARY/raw/IMDB-BINARY'
      elif self.params["dataset"] == "DD":
          self.data_dir = "data/DD/raw/DD"
          
      datareader = DataReader(data_dir=self.data_dir,
                          rnd_state=np.random.RandomState(self.params["seed"]),
                          folds=self.params["n_folds"],           
                          use_cont_node_attr=False,
                          random_walk=self.params["random_walk"],
                          num_walk=self.params["num_walk"],
                          walk_length=self.params["walk_length"],
                          p=self.params["p"],
                          q=self.params["q"],
                          node2vec_hidden=self.params["hidden_agg_lay_size"]
                          )
      
      return datareader
    
    def get_model(self):
        
      # Build graph classification model
      if self.params["model_type"] == 'GCN':
          
        model = GCN(n_feat=self.datareader.data['features_dim'],
                n_class=self.datareader.data['n_classes'],
                n_layer=self.params['num_agg_layer'],
                agg_hidden=self.params['hidden_agg_lay_size'],
                fc_hidden=self.params['fc_hidden_size'],
                dropout=self.params['dropout'],
                pool_type=self.params['pooling_type'],
                device=self.device).to(self.device)
        
      elif self.params["model_type"] == 'GAT':
          
        model = GAT(n_feat=self.datareader.data['features_dim'],
                n_class=self.datareader.data['n_classes'],
                n_layer=self.params['num_agg_layer'],
                agg_hidden=self.params['hidden_agg_lay_size'],
                fc_hidden=self.params['fc_hidden_size'],
                dropout=self.params['dropout'],
                pool_type=self.params['pooling_type'],
                device=self.device).to(self.device)
          
      elif self.params["model_type"] == 'GraphDenseNet':
          
        model = GraphDenseNet(n_feat=self.datareader.data['features_dim'],
                n_class=self.datareader.data['n_classes'],
                n_layer=self.params['num_agg_layer'],
                agg_hidden=self.params['hidden_agg_lay_size'],
                fc_hidden=self.params['fc_hidden_size'],
                dropout=self.params['dropout'],
                pool_type=self.params["pooling_type"],
                device=self.device).to(self.device)
          
      return model


    def loaders_train_test_setup(self, fold_id):
      
      # STEP 1: create test and train fold, get train and test loaders 

      print(f"Fold number: {fold_id}")

      loaders = []

      for split in ['train', 'test']:

        # Build GDATA object

        if split == 'train':
          gdata = GraphData(fold_id=fold_id,
                            datareader=self.datareader,
                            split=split,
                            random_walk=self.params["random_walk"],
                            n_graph_subsampling=self.params["n_graph_subsampling"],
                            graph_node_subsampling=self.params["graph_node_subsampling"],
                            graph_subsampling_rate=self.params["graph_subsampling_rate"])
          
        else:
  
          gdata = GraphData(fold_id=fold_id,
                            datareader=self.datareader,
                            split=split,
                            random_walk=self.params["random_walk"],
                            n_graph_subsampling=0,
                            graph_node_subsampling=self.params["graph_node_subsampling"],
                            graph_subsampling_rate=self.params["graph_subsampling_rate"])  
      
        # Build graph data pytorch loader
        loader = torch.utils.data.DataLoader(gdata, 
                                            batch_size=self.params["batch_size"],
                                            shuffle=split.find('train') >= 0,
                                            num_workers=self.params["threads"],
                                            drop_last=False)
        loaders.append(loader)
      
      # Total trainable param
      c = 0
      for p in filter(lambda p: p.requires_grad, self.model.parameters()):
          c += p.numel()
      print('N trainable parameters:', c)

      optimizer = optim.Adam(
                      filter(lambda p: p.requires_grad, self.model.parameters()),
                      lr=self.params["lr"],
                      weight_decay=self.params["weight_decay"],
                      betas=(0.5, 0.999))
  
      scheduler = lr_scheduler.MultiStepLR(optimizer, [20, 30], gamma=0.1)
        
      return loaders, optimizer, scheduler

    def train(self, train_loader, optimizer, scheduler, epoch):

      total_time_iter = 0
      self.model.train()
      start = time.time()
      train_loss, n_samples = 0, 0
      for batch_idx, data in enumerate(train_loader):
          for i in range(len(data)):
              data[i] = data[i].to(self.device)
          optimizer.zero_grad()
          output = self.model(data)
          loss = self.loss_fn(output, data[4])
          loss.backward()
          optimizer.step()
          time_iter = time.time() - start
          total_time_iter += time_iter
          train_loss += loss.item() * len(output)
          n_samples += len(output)
          if batch_idx % self.params["print_logger"] == 0 or batch_idx == len(train_loader) - 1:
              print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} (avg: {:.6f}) \tsec/iter: {:.4f}'.format(
                  epoch, n_samples, len(train_loader.dataset),
                  100. * (batch_idx + 1) / len(train_loader), loss.item(), train_loss / n_samples, time_iter / (batch_idx + 1) ))
      scheduler.step()
      return total_time_iter / (batch_idx + 1)
      

    def test(self, test_loader, epoch):

      print('Test model ...')

      self.model.eval()
      test_loss, correct, n_samples = 0, 0, 0

      for batch_idx, data in enumerate(test_loader):
          for i in range(len(data)):
            data[i] = data[i].to(self.device)
          
          output = self.model(data)
          loss = self.loss_fn(output, data[4], reduction='sum')
          test_loss += loss.item()
          n_samples += len(output)
          pred = output.detach().cpu().max(1, keepdim=True)[1]

          correct += pred.eq(data[4].detach().cpu().view_as(pred)).sum().item()

      test_loss /= n_samples

      acc = 100. * correct / n_samples

      print('Test set (epoch {}): Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(epoch, 
                                                                                            test_loss, 
                                                                                            correct, 
                                                                                            n_samples, acc))
      
      return acc
            
            
    def fit(self): 

      for fold_id in range(self.params["n_folds"]):
         
        loaders, optimizer, scheduler = self.loaders_train_test_setup(fold_id)

        total_time = 0

        for epoch in range(self.params["epochs"]):
            total_time_iter = self.train(loaders[0], optimizer, scheduler, epoch)
            total_time += total_time_iter
            acc = self.test(self.loaders[1], epoch)

        self.acc_folds.append(round(acc,2))

        self.time_folds.append(round(total_time/self.params["epochs"],2))
          
      print(self.acc_folds)
      print('{}-fold cross validation avg acc (+- std): {} ({})'.format(self.params["n_folds"], statistics.mean(self.acc_folds), statistics.stdev(self.acc_folds)))
      
      result_list = []
      result_list.append(self.params["dataset"])
      result_list.append(self.params["dataset"])

      for acc_fold in self.acc_folds:
        result_list.append(str(acc_fold))

      result_list.append(statistics.mean(self.acc_folds))
      result_list.append(statistics.stdev(self.acc_folds))
      result_list.append(statistics.mean(self.time_folds))
      