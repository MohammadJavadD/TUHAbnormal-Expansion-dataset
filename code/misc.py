from typing import Callable

import pandas as pd
import torch
import torch.utils.data
import torchvision
import torch
from torch import nn

# Authors: Lukas Gemein <l.gemein@gmail.com>
#          Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import pandas as pd
from mne_bids import BIDSPath, read_raw_bids
import mne
import os

# from code.nmt_tuh_load_pp import N_JOBS
mne.set_log_level('ERROR') 
from braindecode.datasets.base import BaseDataset, BaseConcatDataset, WindowsDataset

def load_beyond_ds(beyond_path, subject_ids, N_JOBS=-1):
        # with io.capture_output() as captured:
    # Load your EEG BIDS dataset using mne-bids

    # bids_root = "../../CAE/BIDS_EXPORT/" # change this to your actual path
    bids_root = beyond_path # change this to your actual path
    # bids_root = train_folder # change this to your actual path
    subject = "1" # change this to your subject ID
    # session = "1" # change this to your subject ID
    task = "unnamed" #"aviation"
    run = "1"
    n_runs = 2
    #_eeg

    # Path to your participants.tsv file
    participants_tsv = os.path.join(bids_root , 'participants.tsv')

    # Read the .tsv file using pandas
    participants_df = pd.read_csv(participants_tsv, delimiter='\t')

    bids_path = BIDSPath(subject=subject, task=task, run=run, root=bids_root)
    bids_paths = [BIDSPath(subject=str(subject), task=task, run=str(run), root=bids_root) for subject in subject_ids for run in range(1,n_runs+1)]
    info = [(bids_path.subject, bids_path.run, gender) for bids_path, gender in zip(bids_paths,participants_df['Gender'].repeat(n_runs))]

    raw = mne.io.read_epochs_eeglab(bids_path)
    sfreq = raw.info['sfreq']
    raws_train = [mne.io.read_epochs_eeglab(bids_path) for bids_path in bids_paths]
    for raw in raws_train:
        raw.resample(sfreq=100)
        raw.events[:,2] = raw.events[:,2] - 1

    # Convert the epochs to braindecode format using braindecode.datautil.signal_target
    from misc import create_from_mne_epochs

    from braindecode.preprocessing import (
        preprocess, Preprocessor, create_fixed_length_windows, scale as multiply)

    # common_ch = sorted(['C4', 'P3', 'F4', 'F8', 'Fp2', 'C3', 'Fz', 'Fp1', 'Cz', 'P4', 'O1', 'O2', 'F3', 'F7', 'Pz'])
    common_ch = sorted(['T7', 'T8', 'C5', 'C6', 'P5', 'P6', 'C4', 'P3', 'F4', 'F8', 'Fp2', 'C3', 'Fz', 'Fp1', 'Cz', 'P4', 'O1', 'O2', 'F3', 'F7', 'Pz'])
    
    preprocessors_beyond = [
        # Preprocessor(standard_scale, channel_wise=True),
        Preprocessor('pick_channels', ch_names=common_ch, ordered=True),
        # Preprocessor('resample', sfreq=100),
        # Preprocessor('filter', l_freq=None, h_freq=high_cut_hz, n_jobs=n_jobs)
    ]

    return raws_train, info, preprocessors_beyond

    # return windows_dataset_Beyond

def create_from_mne_epochs(list_of_epochs, info, window_size_samples,
                           window_stride_samples, drop_last_window):
    """Create WindowsDatasets from mne.Epochs

    Parameters
    ----------
    list_of_epochs: array-like
        list of mne.Epochs
    window_size_samples: int
        window size
    window_stride_samples: int
        stride between windows
    drop_last_window: bool
        whether or not have a last overlapping window, when
        windows do not equally divide the continuous signal

    Returns
    -------
    windows_datasets: BaseConcatDataset
        X and y transformed to a dataset format that is compativle with skorch
        and braindecode
    """
    # Prevent circular import
    from braindecode.preprocessing.windowers import _check_windowing_arguments
    _check_windowing_arguments(0, 0, window_size_samples,
                               window_stride_samples)

    list_of_windows_ds = []
    for epochs_i, epochs in enumerate(list_of_epochs):
        event_descriptions = epochs.events[:, 2]
        original_trial_starts = epochs.events[:, 0]
        stop = len(epochs.times) - window_size_samples

        # already includes last incomplete window start
        starts = np.arange(0, stop + 1, window_stride_samples)

        if not drop_last_window and starts[-1] < stop:
            # if last window does not end at trial stop, make it stop there
            starts = np.append(starts, stop)

        fake_events = [[start, window_size_samples, -1] for start in
                       starts]

        for trial_i, trial in enumerate(epochs):
            metadata = pd.DataFrame({
                'i_window_in_trial': np.arange(len(fake_events)),
                'i_start_in_trial': starts + original_trial_starts[trial_i],
                'i_stop_in_trial': starts + original_trial_starts[
                    trial_i] + window_size_samples,
                'target': len(fake_events) * [event_descriptions[trial_i]],
                'subject':info[trial_i][0],
                'run':info[trial_i][1],
                'gender':info[trial_i][2],
                 'trial':trial_i,
            })
            # window size - 1, since tmax is inclusive
            mne_epochs = mne.Epochs(
                mne.io.RawArray(trial, epochs.info), fake_events,
                baseline=None,
                tmin=0,
                tmax=(window_size_samples - 1) / epochs.info["sfreq"],
                metadata=metadata)

            mne_epochs.drop_bad(reject=None, flat=None)

            windows_ds = WindowsDataset(mne_epochs, description={'subject':info[epochs_i][0],'run':info[epochs_i][1], 'trial':trial_i, 'gender':info[epochs_i][2]})
            list_of_windows_ds.append(windows_ds)

    return BaseConcatDataset(list_of_windows_ds)

class CroppedLoss_sd(nn.Module):
    """Compute Loss after averaging predictions across time.
    Assumes predictions are in shape:
    n_batch size x n_classes x n_predictions (in time)"""

    def __init__(self, loss_function):
        super().__init__()
        self.loss_function = loss_function
        self.sd_reg = 0.0

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
        avg_preds = avg_preds.squeeze(dim=1)
        penalty = (avg_preds ** 2).mean()

        return self.loss_function(avg_preds, targets) + self.sd_reg * penalty

def defualt_parser(args):
    if args.model_name == 'EEGNet':
        args.drop_prob=0.25
        args.lr=0.001
        args.weight_decay=0
    elif args.model_name == 'Deep4Net':
        args.drop_prob=0.5
        args.lr=0.01
        args.weight_decay=0.0005
    elif  args.model_name == 'Shallow':
        args.drop_prob=0.5
        args.lr=0.000625
        args.weight_decay=0
    elif args.model_name == 'TCN':
        args.drop_prob=0.05270154233150525
        args.lr=0.0011261049710243193
        args.weight_decay=5.83730537673086e-07
    elif args.model_name == 'TCN_var':
        args.drop_prob=0.05270154233150525
        args.lr=0.0011261049710243193
        args.weight_decay=5.83730537673086e-07
    else:
        print("model name not found")
        


    return args
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

        print("label_to_count:", label_to_count)

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
from braindecode.augmentation import Transform



def return_fft_f(X, y):
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
    X = torch.abs(torch.fft.fft(X, dim=-1))

    return X, y

class return_fft(Transform):
    """scale each input with a given probability.

    Parameters
    ----------
    probability : float
        Float setting the probability of applying the operation.
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Used to decide whether or not to transform given the probability
        argument. Defaults to None.
    """
    operation = staticmethod(return_fft_f)

    def __init__(
        self,
        probability,
        random_state=None
    ):
        super().__init__(
            probability=probability,
            random_state=random_state
        )



def scale_01_f(X, y):
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
    # print(X.shape, X, y, X.min(), X.max())
    # min = X.min()
    # print(min.shape)
    # exit()
    # X = (X-X.min())/(X.max()-X.min()) #
    # X = (X-X.min(-1, keepdim=True)[0])/ (X.max(-1, keepdim=True)[0]-X.min(-1, keepdim=True)[0])
    X = 2 * (X - X.min(-1, keepdim=True)[0]) / (X.max(-1, keepdim=True)[0] - X.min(-1, keepdim=True)[0]) - 1
    # for ii in range(X.shape[0]):
    #     X[ii,:] = (X[ii,:] - X[ii,:].min()) / (X[ii,:].max() - X[ii,:].min())
    # # # X = normalize (X.unsqueeze(dim=3),mean, std, inplace=False).squeeze()

    return X, y

class scale_01(Transform):
    """scale each input with a given probability.

    Parameters
    ----------
    probability : float
        Float setting the probability of applying the operation.
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Used to decide whether or not to transform given the probability
        argument. Defaults to None.
    """
    operation = staticmethod(scale_01_f)

    def __init__(
        self,
        probability,
        random_state=None
    ):
        super().__init__(
            probability=probability,
            random_state=random_state
        )

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
    
class CroppedLoss_coral(nn.Module):
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
    

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn([8, 15, 3000])
    x = x.to(device)
    print(x.shape, x.device)
    # x = torch.unsqueeze(x, dim=1)
    print(x.shape)
    cfg = 10
    # model = Sim_CNN(cfg)
    model = sim_gpt(cfg)
    model.to(device)
    y = model(x)
    print(y.shape)