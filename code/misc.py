from typing import Callable

import pandas as pd
import torch
import torch.utils.data
import torchvision


class ImbalancedDatasetSampler_with_ds(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset

    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(
        self,
        dataset,
        labels: list = None,
        dataset_label: list = None,
        indices: list = None,
        num_samples: int = None,
        callback_get_label: Callable = None,
    ):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset) if labels is None else labels
        # df.index = self.indices

        # distribution of datasets in the dataset
        df["dataset_label"] = self._get_labels(dataset) if labels is None else dataset_label

        # 
        df.index = self.indices
        df = df.sort_index()
        
        # print("df:", df)
        label_to_count = df[['label', 'dataset_label']].value_counts().to_list()
        # label_to_count = df["label"].value_counts()
        print("label_to_count:", label_to_count)

        # weights = 1.0 / label_to_count[df["label"]]
        weights = []
        for i in range(len(df["label"])):
            if df["label"][i] == 0 and df["dataset_label"][i] == 'Tuh':
                weights.append(1 / len(df[(df['label']==0) & (df['dataset_label']=='Tuh')]))
            elif df["label"][i] == 1 and df["dataset_label"][i] == 'Tuh':
                weights.append(1 / len(df[(df['label']==1) & (df['dataset_label']=='Tuh')]))
            elif df["label"][i] == 0 and df["dataset_label"][i] == 'Nmt':
                weights.append(1 / len(df[(df['label']==0) & (df['dataset_label']=='Nmt')]))
            elif df["label"][i] == 1 and df["dataset_label"][i] == 'Nmt':
                weights.append(1 / len(df[(df['label']==1) & (df['dataset_label']=='Nmt')]))

        # print("weights", weights)

        self.weights = torch.DoubleTensor(weights)

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torch.utils.data.TensorDataset):
            return dataset.tensors[1]
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels.tolist()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torchvision.datasets.DatasetFolder):
            return dataset.samples[:][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[:][1]
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.get_labels()
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset

    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(
        self,
        dataset,
        labels: list = None,
        indices: list = None,
        num_samples: int = None,
        callback_get_label: Callable = None,
    ):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset) if labels is None else labels
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torch.utils.data.TensorDataset):
            return dataset.tensors[1]
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels.tolist()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torchvision.datasets.DatasetFolder):
            return dataset.samples[:][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[:][1]
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.get_labels()
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples



import math
import copy
from dataclasses import dataclass


import torch
from torch import nn
from torchvision import models
from torch.nn import functional as F


import numpy as np
import matplotlib.pyplot as plt

from braindecode.models import ShallowFBCSPNet, Deep4Net, EEGResNet, EEGNetv4, TCN

from torch.optim import swa_utils
from skorch.callbacks import Callback

from braindecode.augmentation import Transform
from torchvision.transforms.functional import normalize

def scale_norm_f(X, y , mean, std):
    """normalize ds to mean and std operation.

    Parameters
    ----------
    X : torch.Tensor
        EEG input example or batch.
    y : torch.Tensor
        EEG labels for the example or batch.

    Returns
    -------
    torch.Tensor
        Transformed inputs.
    torch.Tensor
        Transformed labels.
    """
    # print(X.shape,mean.shape)
    for ii in range(X.shape[0]):
        for jj in range(X.shape[1]):
            X[ii,jj,:] = (X[ii,jj,:] - mean[jj])/std[jj]
    # # X = normalize (X.unsqueeze(dim=3),mean, std, inplace=False).squeeze()

    return X, y

class scale_norm(Transform):
    """Flip the sign axis of each input with a given probability.

    Parameters
    ----------
    probability : float
        Float setting the probability of applying the operation.
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Used to decide whether or not to transform given the probability
        argument. Defaults to None.
    """
    operation = staticmethod(scale_norm_f)

    def __init__(
        self,
        probability,
        mean,
        std,
        random_state=None
    ):
        super().__init__(
            probability=probability,
            random_state=random_state
        )
        self.mean = mean
        self.std = std

    def get_augmentation_params(self, *batch):
        """Return transform parameters.

        Parameters
        ----------
        X : tensor.Tensor
            The data.
        y : tensor.Tensor
            The labels.

        Returns
        -------
        params : dict
            Contains:

            * phase_noise_magnitude : float
                The magnitude of the transformation.
            * random_state : numpy.random.Generator
                The generator to use.
        """
        return {
            "mean": self.mean,
            "std": self.std
            }

class StochasticWeightAveraging(Callback):
    def __init__(
            self,
            swa_utils,
            swa_start=10,
            verbose=0,
            sink=print,
            **kwargs  # additional arguments to swa_utils.SWALR
    ):
        self.swa_utils = swa_utils
        self.swa_start = swa_start
        self.verbose = verbose
        self.sink = sink
        vars(self).update(kwargs)

    @property
    def kwargs(self):
        # These are the parameters that are passed to SWALR.
        # Parameters that don't belong there must be excluded.
        excluded = {'swa_utils', 'swa_start', 'verbose', 'sink'}
        kwargs = {key: val for key, val in vars(self).items()
                  if not (key in excluded or key.endswith('_'))}
        return kwargs

    def on_train_begin(self, net, **kwargs):
        self.optimizer_swa_ = self.swa_utils.SWALR(net.optimizer_, **self.kwargs)
        if not hasattr(net, 'module_swa_'):
            # net.module_swa_ = self.swa_utils.AveragedModel(net.module_)
            with net._current_init_context('module'):
                net.module_swa_ = self.swa_utils.AveragedModel(net.module_)
            
    def on_epoch_begin(self, net, **kwargs):
        if self.verbose and len(net.history) == self.swa_start + 1:
            self.sink("Using SWA to update parameters")

    def on_epoch_end(self, net, **kwargs):
        if len(net.history) >= self.swa_start + 1:
            net.module_swa_.update_parameters(net.module_)
            self.optimizer_swa_.step()

    def on_train_end(self, net, X, y=None, **kwargs):
        if self.verbose:
            self.sink("Using training data to update batch norm statistics of the SWA model")

        loader = net.get_iterator(net.get_dataset(X, y))
        self.swa_utils.update_bn(loader, net.module_swa_, device = net.device)


class CroppedLoss_org(nn.Module):
    """Compute Loss after averaging predictions across time.
    Assumes predictions are in shape:
    n_batch size x n_classes x n_predictions (in time)"""

    def __init__(self, loss_function):
        super().__init__()
        self.loss_function = loss_function

    def forward(self, preds, targets):
        """Forward pass.

        Parameters
        ----------
        preds: torch.Tensor
            Model's prediction with shape (batch_size, n_classes, n_times).
        targets: torch.Tensor
            Target labels with shape (batch_size, n_classes, n_times).
        """
        avg_preds = torch.mean(preds, dim=2)
        # avg_preds = torch.median(preds, dim=2).values
        # print("avg_preds:", avg_preds)
        avg_preds = avg_preds.squeeze(dim=1)
        return self.loss_function(avg_preds, targets)
    
class CroppedLoss_sd(nn.Module):
    """Compute Loss after averaging predictions across time.
    Assumes predictions are in shape:
    n_batch size x n_classes x n_predictions (in time)"""

    def __init__(self, loss_function):
        super().__init__()
        self.loss_function = loss_function

    def forward(self, preds, targets):
        """Forward pass.

        Parameters
        ----------
        preds: torch.Tensor
            Model's prediction with shape (batch_size, n_classes, n_times).
        targets: torch.Tensor
            Target labels with shape (batch_size, n_classes, n_times).
        """
        # print("preds:", preds.shape)
        # print("targets:", targets)
        meta_info = targets

        if isinstance(targets, list):
            targets = meta_info[0]

            a=preds[meta_info[1]==0]
            a=a.reshape(a.shape[0],a.shape[1]*a.shape[2])
            b=preds[meta_info[1]==1]
            b=b.reshape(b.shape[0],b.shape[1]*b.shape[2])
            dis = self.dis_coral(a,b)
            # print("dis:", dis)
        else:
            dis = 0

        avg_preds = torch.mean(preds, dim=2)
        avg_preds = avg_preds.squeeze(dim=1)

        ## sd penalty
        self.sd_reg = 5.5 * 10 ** -8
        # penalty = (preds.exp() ** 2).mean()
        # penalty = (avg_preds.exp() ** 2).mean()
        penalty = dis
        # print('penalty:', penalty)
        objective = self.loss_function(avg_preds, targets) #+ self.sd_reg * penalty
        return objective

    def dis_coral(self,x,y):
        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x
        cent_y = y - mean_y
        cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
        cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()

        return mean_diff + cova_diff 
    
class new_ds(torch.utils.data.Dataset):
    def __init__(self, external_dataset):
        self.external_dataset = external_dataset
        self.targets = self.external_dataset.get_metadata().target
        self.domains = self.external_dataset.get_metadata().dataset
        self.domains = self.domains.replace('Tuh', 0).replace('Nmt', 1)

    def __len__(self):
        return len(self.external_dataset)

    def __getitem__(self, i):
        Xi, yi, indi = self.external_dataset[i]
        # print(i, Xi.shape, yi, indi)
        # print(self.external_dataset.get_metadata().target.iloc[i])
        yi_new = self.targets.iloc[i]
        # print(self.external_dataset.get_metadata().dataset.iloc[i])
        di = self.domains.iloc[i]
        return Xi, (yi, di), indi #yi