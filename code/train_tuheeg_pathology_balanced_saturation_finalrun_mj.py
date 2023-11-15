import time
import os
import mne
from sympy import N

# from code.nmt_tuh_load_pp import N_JOBS
mne.set_log_level('ERROR')

from skorch.callbacks import WandbLogger


from warnings import filterwarnings
filterwarnings('ignore')


from IPython.utils import io

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import csv


import torch
from torch.nn.functional import relu
from torch.utils.data import WeightedRandomSampler

from braindecode import EEGClassifier
from braindecode.training.losses import CroppedLoss
CroppedLoss.reduction = "mean"
from misc import CroppedLoss_sd
from misc import StochasticWeightAveraging, load_beyond_ds
CroppedLoss_sd.reduction = "mean"

from skorch.scoring import loss_scoring
from braindecode.models import Deep4Net,ShallowFBCSPNet,EEGNetv4, TCN
from braindecode.util import set_random_seeds
from braindecode.models.util import to_dense_prediction_model, get_output_shape
from braindecode.datasets import BaseConcatDataset

from braindecode.datautil.windowers import create_fixed_length_windows
from braindecode.datautil.serialization import  load_concat_dataset

from braindecode.datasets import BaseConcatDataset
from braindecode.datautil.preprocess import preprocess, Preprocessor, exponential_moving_standardize
# from braindecode.augmentation import Mixup, Mixup_class
from torchvision.transforms import Normalize
from braindecode.augmentation import AugmentedDataLoader, SignFlip, IdentityTransform, ChannelsDropout, FrequencyShift, ChannelsShuffle, SmoothTimeMask, BandstopFilter, GaussianNoise


from braindecode.training import trial_preds_from_window_preds

from braindecode.augmentation import Transform
import random 

from functools import partial 
from skorch.callbacks import LRScheduler, EarlyStopping,Checkpoint, EpochScoring
from skorch.helper import predefined_split


    
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error

import sys
sys.path.insert(0, '/home/kiessnek/TUHEEG_decoding/code/')
from misc import ImbalancedDatasetSampler, ImbalancedDatasetSampler_with_ds
# from ExtendedAdam import ExtendedAdam

from itertools import product
from functools import partial 

#import GPUtil


#######################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm



def plot_confusion_matrix_paper(confusion_mat, p_val_vs_a=None,
                                p_val_vs_b=None,
                                class_names=None, figsize=None,
                                colormap=cm.bwr,
                                textcolor='black', vmin=None, vmax=None,
                                fontweight='normal',
                                rotate_row_labels=90,
                                rotate_col_labels=0,
                                with_f1_score=False,
                                norm_axes=(0, 1),
                                rotate_precision=False,
                                class_names_fontsize=12,
                                ):
    
    # TODELAY: split into several functions
    # transpose to get confusion matrix same way as matlab
    confusion_mat = confusion_mat.T
    # then have to transpose pvals also
    if p_val_vs_a is not None:
        p_val_vs_a = p_val_vs_a.T
    if p_val_vs_b is not None:
        p_val_vs_b = p_val_vs_b.T
    n_classes = confusion_mat.shape[0]
    if class_names is None:
        class_names = [str(i_class + 1) for i_class in range(n_classes)]

    # norm by number of targets (targets are columns after transpose!)
    # normed_conf_mat = confusion_mat / np.sum(confusion_mat,
    #    axis=0).astype(float)
    # norm by all targets
    normed_conf_mat = confusion_mat / np.float32(np.sum(confusion_mat, axis=norm_axes, keepdims=True))

    fig = plt.figure(figsize=figsize)
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    if vmin is None:
        vmin = np.min(normed_conf_mat)
    if vmax is None:
        vmax = np.max(normed_conf_mat)

    # see http://stackoverflow.com/a/31397438/1469195
    # brighten so that black text remains readable
    # used alpha=0.6 before
    def _brighten(x, ):
        brightened_x = 1 - ((1 - np.array(x)) * 0.4)
        return brightened_x

    brightened_cmap = _cmap_map(_brighten, colormap) #colormap #
    ax.imshow(np.array(normed_conf_mat), cmap=brightened_cmap,
              interpolation='nearest', vmin=vmin, vmax=vmax)

    # make space for precision and sensitivity
    plt.xlim(-0.5, normed_conf_mat.shape[0]+0.5)
    plt.ylim(normed_conf_mat.shape[1] + 0.5, -0.5)
    width = len(confusion_mat)
    height = len(confusion_mat[0])
    for x in range(width):
        for y in range(height):
            if x == y:
                this_font_weight = 'bold'
            else:
                this_font_weight = fontweight
            annotate_str = "{:d}".format(confusion_mat[x][y])
            if p_val_vs_a is not None and p_val_vs_a[x][y] < 0.05:
                annotate_str += " *"
            else:
                annotate_str += "  "
            if p_val_vs_a is not None and p_val_vs_a[x][y] < 0.01:
                annotate_str += u"*"
            if p_val_vs_a is not None and p_val_vs_a[x][y] < 0.001:
                annotate_str += u"*"

            if p_val_vs_b is not None and p_val_vs_b[x][y] < 0.05:
                annotate_str += u" ◊"
            if p_val_vs_b is not None and p_val_vs_b[x][y] < 0.01:
                annotate_str += u"◊"
            if p_val_vs_b is not None and p_val_vs_b[x][y] < 0.001:
                annotate_str += u"◊"
            annotate_str += "\n"
            ax.annotate(annotate_str.format(confusion_mat[x][y]),
                        xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center', fontsize=12,
                        color=textcolor,
                        fontweight=this_font_weight)
            if x != y or (not with_f1_score):
                ax.annotate(
                    "\n\n{:4.1f}%".format(
                        normed_conf_mat[x][y] * 100),
                    #(confusion_mat[x][y] / float(np.sum(confusion_mat))) * 100),
                    xy=(y, x),
                    horizontalalignment='center',
                    verticalalignment='center', fontsize=10,
                    color=textcolor,
                    fontweight=this_font_weight)
            else:
                assert x == y
                precision = confusion_mat[x][x] / float(np.sum(
                    confusion_mat[x, :]))
                sensitivity = confusion_mat[x][x] / float(np.sum(
                    confusion_mat[:, y]))
                f1_score = 2 * precision * sensitivity / (precision + sensitivity)

                ax.annotate("\n{:4.1f}%\n{:4.1f}% (F)".format(
                    (confusion_mat[x][y] / float(np.sum(confusion_mat))) * 100,
                    f1_score * 100),
                    xy=(y, x + 0.1),
                    horizontalalignment='center',
                    verticalalignment='center', fontsize=10,
                    color=textcolor,
                    fontweight=this_font_weight)

    # Add values for target correctness etc.
    for x in range(width):
        y = len(confusion_mat)
        correctness = confusion_mat[x][x] / float(np.sum(confusion_mat[x, :]))
        annotate_str = ""
        if p_val_vs_a is not None and p_val_vs_a[x][y] < 0.05:
            annotate_str += " *"
        else:
            annotate_str += "  "
        if p_val_vs_a is not None and p_val_vs_a[x][y] < 0.01:
            annotate_str += u"*"
        if p_val_vs_a is not None and p_val_vs_a[x][y] < 0.001:
            annotate_str += u"*"

        if p_val_vs_b is not None and p_val_vs_b[x][y] < 0.05:
            annotate_str += u" ◊"
        if p_val_vs_b is not None and p_val_vs_b[x][y] < 0.01:
            annotate_str += u"◊"
        if p_val_vs_b is not None and p_val_vs_b[x][y] < 0.001:
            annotate_str += u"◊"
        annotate_str += "\n{:5.2f}%".format(correctness * 100)
        ax.annotate(annotate_str,
                    xy=(y, x),
                    horizontalalignment='center',
                    verticalalignment='center', fontsize=12)

    for y in range(height):
        x = len(confusion_mat)
        correctness = confusion_mat[y][y] / float(np.sum(confusion_mat[:, y]))
        annotate_str = ""
        if p_val_vs_a is not None and p_val_vs_a[x][y] < 0.05:
            annotate_str += " *"
        else:
            annotate_str += "  "
        if p_val_vs_a is not None and p_val_vs_a[x][y] < 0.01:
            annotate_str += u"*"
        if p_val_vs_a is not None and p_val_vs_a[x][y] < 0.001:
            annotate_str += u"*"

        if p_val_vs_b is not None and p_val_vs_b[x][y] < 0.05:
            annotate_str += u" ◊"
        if p_val_vs_b is not None and p_val_vs_b[x][y] < 0.01:
            annotate_str += u"◊"
        if p_val_vs_b is not None and p_val_vs_b[x][y] < 0.001:
            annotate_str += u"◊"
        annotate_str += "\n{:5.2f}%".format(correctness * 100)
        ax.annotate(annotate_str,
                    xy=(y, x),
                    horizontalalignment='center',
                    verticalalignment='center', fontsize=12)

    overall_correctness = np.sum(np.diag(confusion_mat)) / np.sum(confusion_mat).astype(float)
    ax.annotate("{:5.2f}%".format(overall_correctness * 100),
                xy=(len(confusion_mat), len(confusion_mat)),
                horizontalalignment='center',
                verticalalignment='center', fontsize=12,
                fontweight='bold')

    plt.xticks(range(width), class_names, fontsize=class_names_fontsize,
               rotation=rotate_col_labels)
    plt.yticks(np.arange(0,height), class_names,
               va='center',
               fontsize=class_names_fontsize, rotation=rotate_row_labels)
    plt.grid(False)
    plt.ylabel('Predictions', fontsize=15)
    plt.xlabel('Targets', fontsize=15)

    # n classes is also shape of matrix/size
    ax.text(-1.2, n_classes+0.2, "Specificity /\nSensitivity", ha='center', va='center',
            fontsize=13)
    if rotate_precision:
        rotation=90
        x_pos = -1.1
        va = 'center'
    else:
        rotation=0
        x_pos = -0.8
        va = 'top'
    ax.text(n_classes, x_pos, "Precision", ha='center', va=va,
            rotation=rotation,  # 270,
            fontsize=13)

    return fig

# see http://stackoverflow.com/a/31397438/1469195
def _cmap_map(function, cmap, name='colormap_mod', N=None, gamma=None):
    """
    Modify a colormap using `function` which must operate on 3-element
    arrays of [r, g, b] values.

    You may specify the number of colors, `N`, and the opacity, `gamma`,
    value of the returned colormap. These values default to the ones in
    the input `cmap`.

    You may also specify a `name` for the colormap, so that it can be
    loaded using plt.get_cmap(name).
    """
    from matplotlib.colors import LinearSegmentedColormap as lsc
    if N is None:
        N = cmap.N
    if gamma is None:
        gamma = cmap._gamma
    cdict = cmap._segmentdata
    # Cast the steps into lists:
    step_dict = {key: list(map(lambda x: x[0], cdict[key])) for key in cdict}
    # Now get the unique steps (first column of the arrays):
    step_dicts = np.array(list(step_dict.values()))
    step_list = np.unique(step_dicts)
    # 'y0', 'y1' are as defined in LinearSegmentedColormap docstring:
    y0 = cmap(step_list)[:, :3]
    y1 = y0.copy()[:, :3]
    # Go back to catch the discontinuities, and place them into y0, y1
    for iclr, key in enumerate(['red', 'green', 'blue']):
        for istp, step in enumerate(step_list):
            try:
                ind = step_dict[key].index(step)
            except ValueError:
                # This step is not in this color
                continue
            y0[istp, iclr] = cdict[key][ind][1]
            y1[istp, iclr] = cdict[key][ind][2]
    # Map the colors to their new values:
    y0 = np.array(list(map(function, y0)))
    y1 = np.array(list(map(function, y1)))
    # Build the new colormap (overwriting step_dict):
    for iclr, clr in enumerate(['red', 'green', 'blue']):
        step_dict[clr] = np.vstack((step_list, y0[:, iclr], y1[:, iclr])).T
    # Remove alpha, otherwise crashes...
    step_dict.pop('alpha', None)
    return lsc(name, step_dict, N=N, gamma=gamma)

###########################
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from skorch.callbacks import EarlyStopping



def plot_roc_curve(fper, tper, save_name):
    plt.clf()
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.savefig(save_name +'_ROC.png',bbox_inches='tight')
    plt.show()
####################################

from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import accuracy_score,balanced_accuracy_score
from skorch.helper import SliceDataset

def save_as_csv(clf, seed, model_name,ids_to_load_train, ids_to_load_train2, b_acc_merge, b_acc_tuh, b_acc_nmt, write = True):
    
    model = clf.module_
    n_of_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    data = [
    # [seed,dataset, Model, Param, N_of_recordings, b_acc],
    [seed, model_name ,n_of_params, ids_to_load_train, ids_to_load_train2, b_acc_merge, b_acc_tuh, b_acc_nmt]
    ]

    if write:
        with open('output_all.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)

def print_single_ds_performance_groups(clf, test_set):
    test_set.set_description({
        "dataset": ['Nmt' if (type(d) == float and np.isnan(d)) else 'Tuh' for d in test_set.description['version']]},overwrite=True)

    test_set_tuh = test_set.split('dataset')['Tuh']
    test_set_nmt = test_set.split('dataset')['Nmt']

    ## create groups based on gender
    test_set_nmt_normal = test_set_nmt.split('pathological')['0']
    test_set_nmt_abnormal = test_set_nmt.split('pathological')['1']

    #here are the groups
    test_set_nmt_normal_male = test_set_nmt_normal.split('gender')['M']
    test_set_nmt_normal_female = test_set_nmt_normal.split('gender')['F']

    test_set_nmt_abnormal_male = test_set_nmt_abnormal.split('gender')['M']
    test_set_nmt_abnormal_female = test_set_nmt_abnormal.split('gender')['F']

    # Define a dictionary to map variable names to their values
    variables = {
        'test_set_nmt_normal_male': test_set_nmt_normal_male,
        'test_set_nmt_normal_female': test_set_nmt_normal_female,
        'test_set_nmt_abnormal_male': test_set_nmt_abnormal_male,
        'test_set_nmt_abnormal_female': test_set_nmt_abnormal_female
    }

    # Evaluate on all groups
    acc_dict = {}
    for var_name, test_set in variables.items():
        test_X = SliceDataset(test_set, idx=0)
        test_y = test_set.get_metadata().target.to_numpy()
        y_true = test_y
        acc = accuracy_score(np.array(y_true), clf.predict(test_X))
        loss = loss_scoring(clf, test_X, test_y)
        # print(f'loss_{var_name}:', loss)
        print(f'Acc_{var_name}:', acc)
        acc_dict[var_name] = acc

    return acc_dict 
    


def print_single_ds_performance(clf, test_set):
    test_set.set_description({
        "dataset": ['Nmt' if (type(d) == float and np.isnan(d)) else 'Tuh' for d in test_set.description['version']]},overwrite=True)

    test_set_tuh = test_set.split('dataset')['Tuh']
    test_set_nmt = test_set.split('dataset')['Nmt']
    
    test_X = SliceDataset(test_set, idx=0)
    test_y = test_set.get_metadata().target.to_numpy()
    y_true =test_y#[:,0]
    b_acc_merge = balanced_accuracy_score(np.array(y_true), clf.predict(test_X))
    loss_merge = loss_scoring (clf,test_X,test_y) 
    print('loss_merge:', loss_merge)
    print('B_acc_merged:', b_acc_merge)

    test_X = SliceDataset(test_set_tuh, idx=0)
    test_y = test_set_tuh.get_metadata().target.to_numpy()
    y_true =test_y#[:,0]
    b_acc_tuh = balanced_accuracy_score(np.array(y_true), clf.predict(test_X))
    loss_tuh = loss_scoring (clf,test_X,test_y) 
    print('loss_tuh:', loss_tuh)
    print('B_acc_tuh:', b_acc_tuh)

    test_X = SliceDataset(test_set_nmt, idx=0)
    test_y = test_set_nmt.get_metadata().target.to_numpy()
    y_true =test_y#[:,0]
    b_acc_nmt = balanced_accuracy_score(np.array(y_true), clf.predict(test_X))
    # import pandas as pd
    # class_stats = pd.Series(test_y).value_counts()
    # print('class_stats:', class_stats)
    loss_nmt = loss_scoring (clf,test_X,test_y) 
    print('loss_nmt:', loss_nmt)
    print('B_acc_nmt:', b_acc_nmt)

    test_X = SliceDataset(test_set_nmt, idx=0)
    y_true = test_set_nmt.get_metadata().target.to_numpy()
    # np.random.shuffle(y_true)#[:,0]
    b_acc_nmt_shuffle = balanced_accuracy_score(np.random.permutation(y_true), clf.predict(test_X))
    print('b_acc_nmt_shuffle_y_true:', b_acc_nmt_shuffle)

    test_X = SliceDataset(test_set_nmt, idx=0)
    y_true = test_set_nmt.get_metadata().target.to_numpy()
    y_pred = clf.predict(test_X)
    # np.random.shuffle(y_pred)#[:,0]
    b_acc_nmt_shuffle = balanced_accuracy_score(np.array(y_true), np.random.permutation(y_pred))
    print('b_acc_nmt_shuffle_y_pred:', b_acc_nmt_shuffle)

    return b_acc_merge, b_acc_tuh, b_acc_nmt, loss_merge, loss_tuh, loss_nmt
    


def print_single_ds_performance_beyond(clf, test_set):
    
    test_X = SliceDataset(test_set, idx=0)
    # test_y = test_set.get_metadata().target.to_numpy()
    test_y = test_set.get_metadata().gender_bool.to_numpy()
    y_true =test_y#[:,0]
    # print('test_X:', test_X.shape)
    # print('test_y:', test_y)
    b_acc_merge = balanced_accuracy_score(np.array(y_true), clf.predict(test_X))
    loss_merge = None #loss_scoring (clf,test_X,test_y) 
    print('loss_merge:', loss_merge)
    print('B_acc_merged:', b_acc_merge)
    b_acc_tuh, b_acc_nmt,loss_tuh,loss_nmt = None, None, None, None

    return b_acc_merge, b_acc_tuh, b_acc_nmt, loss_merge, loss_tuh, loss_nmt

def keep_unique (ds):
    # keep only unique subjects
    df = ds.description

    subjects = df['subject']
    unique_subjects = df['subject'].unique()
    print(f'There are {len(subjects)} subjects in the dataset')
    print(f'There are {len(unique_subjects)} unique subjects in the dataset')

    # Create a new column 'status' that marks duplicates as 'duplicate' and unique values as 'unique'
    df['status'] = df.duplicated('subject')
    df['status'] = df['status'].replace({True: 'duplicate', False: 'unique'})

    ds.set_description(df, overwrite=True)

    # Keep only one record per subject
    ds = ds.split('status')['unique']
    return ds

def remove_common (ds):
    # Remove noisy labbled subjects
    df = ds.description
    # Split the dataframe by pathological column
    groups = df.groupby('gender_bool')

    # Access each group as a dataframe
    df_yes = groups.get_group(True)
    df_no = groups.get_group(False)

    # Display the first five rows of each group
    df_yes.head()
    df_no.head()

    # Get the unique subjects in each group
    subjects_yes = set(df_yes['subject'].unique())
    subjects_no = set(df_no['subject'].unique())

    # Find the intersection of the two sets
    common_subjects = subjects_yes.intersection(subjects_no)

    # Display the common subjects
    print(f'there are {len(common_subjects)} common subjects in the two pathological groups')

    # Filter the dataframe by the common subjects
    df_common = df[df['subject'].isin(common_subjects)]

    # Display the number of rows of the filtered dataframe
    print(f'There are {len(df_common)} out of {len(df)} rows have their status changed')

    # Create a new column named 'common' with boolean values
    df = df.assign(common=df['subject'].isin(common_subjects))

    # Display the first five rows of the dataframe
    # df.head()
    ds.set_description(df, overwrite=True)

    ds = ds.split('common')['False']
    
    return ds
## end of remove noisy labbled subjects


def train_TUHEEG_pathology(
                    model_name,
                    target_name,
                     drop_prob,
                     batch_size,
                     lr,
                     n_epochs,
                     weight_decay,
                     result_folder,
                     train_folder,
                    #  eval_folder,
                     task_name,
                     ids_to_load_train,
                     seed,
                     cuda = True,
                    pre_trained = False,
                    load_path = None,
                    train_folder2 = None,
                    train_folder3 = None,
                    train_folder4 = None,
                    augment = False,
                    n_tcn_blocks = 5,
                    n_tcn_filters = 55,
                    ids_to_load_train2=None,
                    ids_to_load_train3=None,
                    ids_to_load_train4=None,
                    wandb_run = None,
                    only_eval = False,
                    N_JOBS=4
                    ):

    ###################################
    # target_name = None  # Comment Lukas our target is not supported by default, set None and add later 
    # add_physician_reports = True

    sfreq = 100  #100
    # n_minutes = 20 #

    # n_max_minutes = n_minutes+1
    
    #Drop_prob = [drop_prob]
    # N_REPETITIONS =1
    ####### MODEL DEFINITION ############
    torch.backends.cudnn.benchmark = True
    ######

    # Set random seed to be able to reproduce results
    set_random_seeds(seed=seed, cuda=cuda)

    
   
    print(task_name)

    print('loading data')
    # load from train
    start = time.time()
    # with io.capture_output() as captured:
    ds_1 = load_concat_dataset(train_folder, preload=False,
                                # target_name=['pathological','age','gender'] ,#)
                                # ids_to_load=range(200)
                                )
    if train_folder2:
        print('loading second train folder')
        ds_2 = load_concat_dataset(train_folder2, preload=False,
                                # target_name=['pathological','age','gender'] ,#)
                                # ids_to_load=range(200)
                                )
        print('merging datasets')
        ds_all = BaseConcatDataset([ds_1, ds_2])
        # print('ds_all:', ds_all.description)
    
    ds_all3 = None
    if train_folder3:
        print('loading third train folder')
        ids_to_load3 = None if ids_to_load_train3 is None else range(ids_to_load_train3)
        ds_all3 = load_concat_dataset(train_folder3, preload=False,
                              # target_name=['pathological','age','gender'] ,#)
                              ids_to_load=ids_to_load3
                              )
        # print('merging datasets')
        # ds_all = BaseConcatDataset([ds_all, ds_all2])
    
    ds_all4 = None
    if train_folder4:
        print('loading fourth train folder')
        raws_train, info, preprocessors_beyond = load_beyond_ds(train_folder4, range(1,ids_to_load_train4), N_JOBS=N_JOBS)
        ds_all4 = 1  
        

    end = time.time()

    print('finished loading preprocessed trainset ' + str(end-start))


    target= target_name
    if target=='pathological':
        print("target is being set to pathological clf")
        target = ds_all.description['pathological'].astype(int)
        for d, y in zip(ds_all.datasets, target):
            d.description['pathological'] = y
            d.target_name = 'pathological'
            d.target = d.description[d.target_name]
        ds_all.set_description(pd.DataFrame([d.description for d in ds_all.datasets]), overwrite=True)
    elif target=='sex':
        print("target is being set to sex clf")
        # convert M F to 01
        df = ds_all.description
        df = df.assign(gender_bool=df['gender'].map(lambda x: 0 if x == 'M' else 1 if x == 'F' else None))
        ds_all.set_description(df, overwrite=True)

        #set target
        target = ds_all.description['gender_bool'].astype(int)
        for d, y in zip(ds_all.datasets, target):
            d.description['gender_bool'] = y
            d.target_name = 'gender_bool'
            d.target = d.description[d.target_name]
        ds_all.set_description(pd.DataFrame([d.description for d in ds_all.datasets]), overwrite=True)

        if ds_all3:
            ## do the same for tueg
            # remove None values from the dataset
            ds_all3_splitted = ds_all3.split('gender')
            ds_all3 = BaseConcatDataset([ds_all3_splitted['M'], ds_all3_splitted['F']])
            # convert M F to 01
            df = ds_all3.description
            df = df.assign(gender_bool=df['gender'].map(lambda x: 0 if x == 'M' else 1 if x == 'F' else None))
            ds_all3.set_description(df, overwrite=True)

            #set target
            target = ds_all3.description['gender_bool'].astype(int)
            for d, y in zip(ds_all3.datasets, target):
                d.description['gender_bool'] = y
                d.target_name = 'gender_bool'
                d.target = d.description[d.target_name]
            ds_all3.set_description(pd.DataFrame([d.description for d in ds_all3.datasets]), overwrite=True)
            print('ds_all3:', ds_all3.description)
            x, y = ds_all3[-1]
            print('x:', x.shape)
            print('y:', y)

            ds_all3 = keep_unique(ds_all3)
            ds_all3 = remove_common(ds_all3)
            print('ds_all3:', ds_all3.description)

        
    # load eval set

    print('loading eval data')
    # load from train
    start = time.time()
    # with io.capture_output() as captured:
        # test_complete = load_concat_dataset(eval_folder, preload=False, ids_to_load=None)
    train_set = ds_all.split('train')['True']
    test_complete = ds_all.split('train')['False']

    end = time.time()

    print('finished loading preprocessed trainset ' + str(end-start))

    # target = test_complete.description['pathological'].astype(int)
    
    # for d, y in zip(test_complete.datasets, target):
    #     d.description['pathological'] = y
    #     d.target_name = 'pathological'
    #     d.target = d.description[d.target_name]
    # test_complete.set_description(pd.DataFrame([d.description for d in test_complete.datasets]), overwrite=True)
    
    ## limit training set
    # print('train_set:', train_set.description['pathological'].astype(int)[:64])

    if train_folder2:
        train_set.set_description({
            "dataset": ['Nmt' if (type(d) == float and np.isnan(d)) else 'Tuh' for d in train_set.description['version']]},overwrite=True)

        # print('train_set_datset', train_set.description)
        train_set_tuh = train_set.split('dataset')['Tuh']
        train_set_nmt = train_set.split('dataset')['Nmt']

        n_of_rec_all = train_set_tuh.description.shape[0]
        n_of_rec_to_pick = ids_to_load_train #n_of_rec_all#n_of_rec_all# n_of_rec_all ##2000
        print('n_of_rec_to_pick', n_of_rec_to_pick, 'n_of_rec_all', n_of_rec_all )
        if ids_to_load_train>0:
            assert n_of_rec_to_pick <= n_of_rec_all
            train_set_tuh = train_set_tuh.split({"subsample":np.random.randint(0, n_of_rec_all, n_of_rec_to_pick).tolist()})["subsample"]

        n_of_rec_all = train_set_nmt.description.shape[0]
        n_of_rec_to_pick = ids_to_load_train2 #n_of_rec_all#n_of_rec_all# n_of_rec_all ##2000
        print('n_of_rec_to_pick', n_of_rec_to_pick, 'n_of_rec_all', n_of_rec_all )
        if ids_to_load_train2>0:
            assert n_of_rec_to_pick <= n_of_rec_all
            train_set_nmt = train_set_nmt.split({"subsample":np.random.randint(0, n_of_rec_all, n_of_rec_to_pick).tolist()})["subsample"]
        
        if ids_to_load_train>0 and ids_to_load_train2>0:
            train_set = BaseConcatDataset([train_set_tuh, train_set_nmt])
        elif ids_to_load_train>0:
            train_set = BaseConcatDataset([train_set_tuh])  
        elif ids_to_load_train2>0:
            train_set = BaseConcatDataset([train_set_nmt])

        # target = train_set.description['pathological'].astype(int)
        # dataset = pd.get_dummies(train_set.description['dataset'])['Tuh'].astype(int)
        # print(target, dataset)
        # for d, y, ds in zip(train_set.datasets, target, dataset):
        #     d.description['pathological'] = y
        #     d.target_name = 'pathological'
        #     d.target = d.description[d.target_name]

        #     d.description['dataset'] = ds
        #     d.target_name = 'dataset'
        #     d.ds = d.description[d.target_name]
        # train_set.set_description(pd.DataFrame([d.description for d in train_set.datasets]), overwrite=True)
    ## end limit training set

    ## split trainset to val/train
    a = train_set.description['train'].values
    # choose 15 percent of the indices randomly
    indices = np.random.choice(a.size, int(a.size * 0.15), replace=False)
    # assign False to those indices
    a[indices] = False

    train_set.set_description({
        "train": [ii for ii in a]},overwrite=True)
    
    train_set_splited = train_set.split('train')
    train_set = train_set_splited['True']
    val_set = train_set_splited['False']
    ## end of split trainset to val/train

    if ds_all3:
        ## add additional columns to ds_all3 description
        df = ds_all3.description
        df = df.assign(train=df['gender'].map(lambda x: True))
        ds_all3.set_description(df, overwrite=True)
        ## split trainset to val/train
        a = ds_all3.description['train'].values
        # choose 15 percent of the indices randomly
        indices = np.random.choice(a.size, int(a.size * 0.15), replace=False)
        # assign False to those indices
        a[indices] = False
        # print('a:', a)

        ds_all3.set_description({
            "train": [ii for ii in a]},overwrite=True)

        ds_all3_splited = ds_all3.split('train')
        ds_all3_train_set = ds_all3_splited['True']
        ds_all3_val_set = ds_all3_splited['False']
        print('ds_all3:', ds_all3_train_set.description)
        ## end of split trainset to val/train
   

    print('model:',model_name)
    if not os.path.exists(result_folder + model_name + '/' + 'seed'+str(seed)):
            os.mkdir(result_folder + model_name + '/' + 'seed'+str(seed))
            print("Directory " , result_folder + model_name + '/' + 'seed'+str(seed) ,  " Created ")
    else:    
        print("Directory " , result_folder + model_name + '/' + 'seed'+str(seed) ,  " already exists")
            

    dirSave = result_folder + model_name + '/' + 'seed'+str(seed) + '/' + task_name    
    save_name =model_name   +'_trained_' +  str(task_name)+ '_'  


   # cv = 'SKF'    
    if not os.path.exists(dirSave):
        os.mkdir(dirSave)
        print("Directory " , dirSave ,  " Created ")
    else:    
        print("Directory " , dirSave ,  " already exists")


    result_path = dirSave  +'/' + save_name 

    n_classes=2
    # Extract number of chans from dataset
    n_chans = 21
    input_window_samples =6000
    if model_name == 'Deep4Net':
        n_start_chans = 25
        final_conv_length = 1
        n_chan_factor = 2
        stride_before_pool = True
       # input_window_samples =6000
        model = Deep4Net(
                    n_chans, n_classes,
                    n_filters_time=n_start_chans,
                    n_filters_spat=n_start_chans,
                    input_window_samples=input_window_samples,
                    n_filters_2=int(n_start_chans * n_chan_factor),
                    n_filters_3=int(n_start_chans * (n_chan_factor ** 2.0)),
                    n_filters_4=int(n_start_chans * (n_chan_factor ** 3.0)),
                    final_conv_length=final_conv_length,
                    stride_before_pool=stride_before_pool,
                    drop_prob=drop_prob)
            # Send model to GPU
        if cuda:
            model.cuda()

        to_dense_prediction_model(model)

        n_preds_per_input = get_output_shape(model, n_chans, input_window_samples)[2]

    elif model_name == 'Shallow':  
        n_start_chans = 40
        final_conv_length = 25
        input_window_samples =6000
        model = ShallowFBCSPNet(n_chans,n_classes,
                                input_window_samples=input_window_samples,
                                n_filters_time=n_start_chans,
                                n_filters_spat=n_start_chans,
                                final_conv_length= final_conv_length,
                                drop_prob=drop_prob)
        # Send model to GPU
        if cuda:
            model.cuda()

        to_dense_prediction_model(model)

        n_preds_per_input = get_output_shape(model, n_chans, input_window_samples)[2]

    elif model_name == 'TCN':  
        n_chan_factor = 2
        stride_before_pool = True
        input_window_samples =6000
        l2_decay = 1.7491630095065614e-08
        gradient_clip = 0.25

        model = TCN(
            n_in_chans=n_chans, n_outputs=n_classes,
            n_filters=55,
            n_blocks=5,
            kernel_size=16,
            drop_prob=drop_prob,
            add_log_softmax=True,
#            n_times=input_window_samples)
)
            # Send model to GPU
        if cuda:
            model.cuda()

        n_preds_per_input = get_output_shape(model, n_chans, input_window_samples)[2]

    elif model_name == 'TCN_var':  
        n_chan_factor = 2
        stride_before_pool = True
        input_window_samples =6000
        l2_decay = 1.7491630095065614e-08
        gradient_clip = 0.25

        model = TCN(
            n_in_chans=n_chans, n_outputs=n_classes,
            n_filters=n_tcn_filters, #was 55 before
            n_blocks=n_tcn_blocks,# was 5 before
            kernel_size=16,
            drop_prob=drop_prob,
            add_log_softmax=True,
            # n_times=input_window_samples
            )

            # Send model to GPU
        if cuda:
            model.cuda()

        n_preds_per_input = get_output_shape(model, n_chans, input_window_samples)[2]
        

 
    elif model_name == 'EEGNet':    

        input_window_samples =6000
        final_conv_length=18
        drop_prob=0.25
        model = EEGNetv4(
            n_chans, n_classes,
            input_window_samples=input_window_samples,
            final_conv_length=final_conv_length,
            drop_prob=drop_prob)
        if cuda:
            model.cuda()

        to_dense_prediction_model(model)

        n_preds_per_input = get_output_shape(model, n_chans, input_window_samples)[2]

    print(model_name + ' model sent to cuda')
    print(model)


    if ds_all4:
        from misc import create_from_mne_epochs
        windows_dataset_Beyond = create_from_mne_epochs(
            raws_train,
            info,
            window_size_samples=input_window_samples,
            window_stride_samples=n_preds_per_input,
            drop_last_window=True,
        )

        windows_dataset_Beyond = preprocess(
            concat_ds=windows_dataset_Beyond,
            preprocessors=preprocessors_beyond,
            n_jobs=N_JOBS,
            # save_dir='./',
            overwrite=True,
        )

        ## do the same for BEYOND
        # remove None values from the dataset
        windows_dataset_Beyond_splitted = windows_dataset_Beyond.split('gender')
        windows_dataset_Beyond = BaseConcatDataset([windows_dataset_Beyond_splitted['M'], windows_dataset_Beyond_splitted['F']])
        # convert M F to 01
        df = windows_dataset_Beyond.description
        df = df.assign(gender_bool=df['gender'].map(lambda x: 0 if x == 'M' else 1 if x == 'F' else None))
        windows_dataset_Beyond.set_description(df, overwrite=True)

        #set target
        target = windows_dataset_Beyond.description['gender_bool'].astype(int)
        for d, y in zip(windows_dataset_Beyond.datasets, target):
            d.description['gender_bool'] = y
            d.target_name = 'gender_bool'
            d.target = d.description[d.target_name]
        windows_dataset_Beyond.set_description(pd.DataFrame([d.description for d in windows_dataset_Beyond.datasets]), overwrite=True)
        print('windows_dataset_Beyond:', windows_dataset_Beyond.description)
        # x, y = windows_dataset_Beyond[-1]
        # print('x:', x.shape)
        # print('y:', y)
    
    if ds_all3:
        with io.capture_output() as captured:
            window_train_set_tueg = create_fixed_length_windows(ds_all3_train_set, 
                                                            start_offset_samples=0,
                                                            stop_offset_samples=None,
                                                            preload=False,
                                                            window_size_samples=input_window_samples,
                                                            window_stride_samples=n_preds_per_input,
                                                            drop_last_window=True,
                                                            n_jobs=N_JOBS,
                                                            # mapping={ False: 0, True: 1 , 'M': 0, 'F': 1 },  # map non-digit targets 'M': 0, 'F': 1,
                                                            )
        with io.capture_output() as captured:
            window_val_set_tueg = create_fixed_length_windows(ds_all3_val_set, 
                                                            start_offset_samples=0,
                                                            stop_offset_samples=None,
                                                            preload=True,
                                                            window_size_samples=input_window_samples,
                                                            window_stride_samples=n_preds_per_input,
                                                            drop_last_window=True,
                                                            # mapping={ False: 0, True: 1 , 'M': 0, 'F': 1 },  # map non-digit targets 'M': 0, 'F': 1,
                                                            )

    with io.capture_output() as captured:
         window_train_set = create_fixed_length_windows(train_set, 
                                                        start_offset_samples=0,
                                                        stop_offset_samples=None,
                                                        preload=True,
                                                        window_size_samples=input_window_samples,
                                                        window_stride_samples=n_preds_per_input,
                                                        drop_last_window=True,)
    with io.capture_output() as captured:
         window_val_set = create_fixed_length_windows(val_set, 
                                                        start_offset_samples=0,
                                                        stop_offset_samples=None,
                                                        preload=True,
                                                        window_size_samples=input_window_samples,
                                                        window_stride_samples=n_preds_per_input,
                                                        drop_last_window=True,)
    with io.capture_output() as captured:
         window_test_set = create_fixed_length_windows(test_complete,
                                                        start_offset_samples=0,
                                                        stop_offset_samples=None,preload=False,
                                                        window_size_samples=input_window_samples,
                                                        window_stride_samples=n_preds_per_input,
                                                        drop_last_window=False,)

    from misc import scale_01
    window_train_set.__setattr__('transform', scale_01(probability=1))
    window_val_set.__setattr__('transform', scale_01(probability=1))
    window_test_set.__setattr__('transform', scale_01(probability=1))
    if ds_all3:
        window_train_set_tueg.__setattr__('transform', scale_01(probability=1))
        window_val_set_tueg.__setattr__('transform', scale_01(probability=1))
    if ds_all4:
        windows_dataset_Beyond.__setattr__('transform', scale_01(probability=1))
        
    # ## limit n_channels to 15 common 
    # common_ch = sorted(['C4', 'P3', 'F4', 'F8', 'Fp2', 'C3', 'Fz', 'Fp1', 'Cz', 'P4', 'O1', 'O2', 'F3', 'F7', 'Pz'])
    # common_ch = sorted(['C4', 'P3', 'F4', 'F8', 'FP2', 'C3', 'FZ', 'FP1', 'CZ', 'P4', 'O1', 'O2', 'F3', 'F7', 'PZ'])
    # from braindecode.preprocessing import (
    #     preprocess, Preprocessor, scale as multiply)
    # preprocessors = [
    #     Preprocessor('pick_channels', ch_names=common_ch, ordered=True),
    # ]
    # window_train_set = preprocess(
    # concat_ds=window_train_set,
    # preprocessors=preprocessors,
    # n_jobs=1,
    # # save_dir='./',
    # overwrite=True,
    # )

    # window_val_set = preprocess(
    # concat_ds=window_val_set,
    # preprocessors=preprocessors,
    # n_jobs=1,
    # # save_dir='./',
    # overwrite=True,
    # )

    # window_test_set = preprocess(
    # concat_ds=window_test_set,
    # preprocessors=preprocessors,
    # n_jobs=1,
    # # save_dir='./',
    # overwrite=True,
    # )
    # ##
    #del  train_set
    if ds_all3:
        print('Female train TUEG ' + str(len(window_train_set_tueg.description[window_train_set_tueg.description['gender_bool']==1])))
        print('Male train TUEG ' + str(len(window_train_set_tueg.description[window_train_set_tueg.description['gender_bool']==0])))
    if ds_all4:
        print('Female train Beyond ' + str(len(windows_dataset_Beyond.description[windows_dataset_Beyond.description['gender_bool']==1])))
        print('Male train Beyond ' + str(len(windows_dataset_Beyond.description[windows_dataset_Beyond.description['gender_bool']==0])))
        
    
    print('abnormal train ' + str(len(window_train_set.description[window_train_set.description['pathological']==1])))
    print('normal train ' + str(len(window_train_set.description[window_train_set.description['pathological']==0])))

    print('abnormal val' + str(len(window_val_set.description[window_val_set.description['pathological']==1])))
    print('normal val ' + str(len(window_val_set.description[window_val_set.description['pathological']==0])))

    print('abnormal test' + str(len(window_test_set.description[window_test_set.description['pathological']==1])))
    print('normal test ' + str(len(window_test_set.description[window_test_set.description['pathological']==0])))

    ## data augmentation ##
    transforms_val = [
        # scale_norm(.1,mean, std),
        IdentityTransform(),
        # scale_norm(1.,mean, std),
    ]


    class Compose_RandAug(Transform):
        """Transform composition.

        Callable class allowing to cast a sequence of Transform objects into a
        single one.

        Parameters
        ----------
        transforms: list
            Sequence of Transforms to be composed.
        """

        def __init__(self, transforms, n=2, m=0.5):
            self.transforms = transforms
            self.n = n
            self.m = m
            super().__init__()

        def forward(self, X, y):
            # randomly choose two transforms from the list
            transforms = random.sample(self.transforms, k=self.n)
            # print('transforms_train', transforms)
            # print(X.shape,"X")
            for transform in transforms:
                X, y = transform(X, y)
            return X, y
        
    from braindecode.augmentation import (FTSurrogate, 
    )

    m = 0.5
    if augment:
        transforms_train_pool = [
            IdentityTransform(),
            SignFlip(probability=.1),
            # ChannelsDropout(probability=.1, p_drop=.2),
            FrequencyShift(probability=.5, sfreq=sfreq, max_delta_freq=2),
            SmoothTimeMask(probability=.1, mask_len_samples=600),
            BandstopFilter(probability=.5, sfreq=sfreq, bandwidth=5),
            # ChannelsShuffle(probability=.1),
            GaussianNoise(probability=.5, std=.2),
            FTSurrogate(probability=.5),
            # Mixup(alpha=.1),
            # Mixup_class(alpha=.5),
            # scale_norm(1.,mean, std),
        ]
        # randomly choose two transforms from the list
        # transforms_train = Compose_RandAug(transforms_train, n=2, m=0.5)
        transforms_train = Compose_RandAug(transforms_train_pool, n=3, m=0.5)
    else:
        transforms_train = transforms_val
    ## end of data augmentation ##

    # print('targets', window_train_set.get_metadata().target[:64])
    # print('target=',window_train_set.get_metadata().target, 'ds=', window_train_set.get_metadata().dataset)
    
    ## SWA parameters ##
    from torch.optim import swa_utils
    ## end of SWA parameters ##


    ## edit clf for multi-target ds ##
    # labels=window_train_set.get_metadata().target, dataset_label=window_train_set.get_metadata().dataset)
    from misc import new_ds
    # window_train_set_new = new_ds(window_train_set)
    # print('new ds created')
    # print(window_train_set_new[-1])
    ## end of edit clf for multi-target ds ##

    ## add checkpoint ##
    from skorch.callbacks import Checkpoint
    import datetime
    now = datetime.datetime.now()
    f_params_name = 'best_model_params_'+str(now)+'.pt'
    print(f_params_name)

    checkpoint = Checkpoint(
        dirname=result_path,
        f_params=f_params_name,
        monitor='valid_balanced_accuracy_best'
      )


    if pre_trained:
        state_dicts = torch.load(load_path)# +'state_dict_2023.pt')
        model.load_state_dict(state_dicts, strict= False)
        print('pre-trained model loaded using pytorch')

        ## freeze layers ##
        freez = False
        if freez:
            for ii, (name, param) in enumerate(model.named_parameters()):
                # if 'temporal_block_0' in name or 'temporal_block_1' in name or 'temporal_block_2' in name or 'temporal_block_3' in name: # or 'temporal_block_5' in name or 'conv_classifier' in name:
                if not 'conv_classifier' in name:
                    param.requires_grad = False
                    print('param:', name, param.requires_grad)

        # clf.load_params(
        #     f_params=load_path +'state_dict_2023.pt')#, f_optimizer= load_path +'opt.pkl', f_history=load_path +'history.json')
        #     # f_params=load_path +'model.pkl')#, f_optimizer= load_path +'opt.pkl', f_history=load_path +'history.json')
        # print('pre-trained model loaded')
        # clf.initialize() # This is important!

    ## edit param groups
    param_groups = [name for name, param in model.named_parameters()]
    # print(param_groups)
    param_groups.reverse()
    prev_group_name = param_groups[0].split('.')[-2]
    optimizer__param_groups = []
    lr = lr
    if pre_trained:
        lr_mult = 0.9 #0.9
    else:
        lr_mult = 1

    for idx, name in enumerate(param_groups):
        # parameter group name
        cur_group_name = name.split('.')[-2]
    
        # update learning rate
        if cur_group_name != prev_group_name:
            lr *= lr_mult
            # if idx<len(param_groups)/2:
            #     lr *= lr_mult
            # else:
            #     lr /= lr_mult
            prev_group_name = cur_group_name

        optimizer__param_groups.append((name, {'lr': lr}))
        # lr = lr * lr_mult
    optimizer__param_groups.reverse()
    print(optimizer__param_groups)
    ## end of param group

    window_test_set.set_description({
        "dataset": ['Nmt' if (type(d) == float and np.isnan(d)) else 'Tuh' for d in window_test_set.description['version']]},overwrite=True)

    test_set_tuh = window_test_set.split('dataset')['Tuh']
    test_set_nmt = window_test_set.split('dataset')['Nmt']
    
    if ds_all3:
        train_split = window_val_set_tueg #test_set_nmt #window_val_set_tueg [Carefull!]
        iterator_train_shuffle = True
        iterator_train_sampler = None #ImbalancedDatasetSampler(window_train_set_tueg, labels=window_train_set_tueg.get_metadata().target),#, dataset_label=window_train_set_tueg.get_metadata().dataset),                    
    elif target_name=='pathological':
        train_split = window_val_set
        # iterator_train__sampler = ImbalancedDatasetSampler(window_train_set, labels=window_train_set.get_metadata().target)
        iterator_train_sampler = ImbalancedDatasetSampler_with_ds(window_train_set, labels=window_train_set.get_metadata().target, dataset_label=window_train_set.get_metadata().dataset)
        iterator_train_shuffle = False
    else:
        train_split = window_val_set
        iterator_train_shuffle = True
        iterator_train_sampler = None

    clf = EEGClassifier(
                    model,
                    cropped=True,
                    # iterator_train=DataLoader,  # This tells EEGClassifier to use a custom DataLoader
                    iterator_train=AugmentedDataLoader,  # This tells EEGClassifier to use a custom DataLoader
                    iterator_train__transforms=transforms_train,  # This sets the augmentations to use
                    # iterator_valid =DataLoader,  # This tells EEGClassifier to use a custom DataLoader
                    iterator_valid =AugmentedDataLoader,  # This tells EEGClassifier to use a custom DataLoader
                    iterator_valid__transforms=transforms_val,  # This sets the augmentations to use
                    criterion=CroppedLoss,
                    # criterion=CroppedLoss_sd,
                    criterion__loss_function=torch.nn.functional.nll_loss,
                    optimizer=torch.optim.AdamW,
                    optimizer__param_groups=optimizer__param_groups,
                    train_split=predefined_split(train_split), 
                    # train_split=predefined_split(window_val_set_tueg),
                    optimizer__lr=lr,
                    optimizer__weight_decay=weight_decay,
                    iterator_train__shuffle=iterator_train_shuffle,
                    iterator_train__sampler = iterator_train_sampler,
                    # iterator_train__sampler = ImbalancedDatasetSampler(window_train_set, labels=window_train_set.get_metadata().target),
                    # iterator_train__sampler = ImbalancedDatasetSampler_with_ds(window_train_set, labels=window_train_set.get_metadata().target, dataset_label=window_train_set.get_metadata().dataset),
                    # iterator_valid__sampler = ImbalancedDatasetSampler_with_ds(window_val_set, labels=window_val_set.get_metadata().target, dataset_label=window_val_set.get_metadata().dataset),
                    # iterator_train__sampler = ImbalancedDatasetSampler(window_train_set_tueg, labels=window_train_set_tueg.get_metadata().target),#, dataset_label=window_train_set_tueg.get_metadata().dataset),                    
                    batch_size=batch_size,
                    callbacks=[
                        # EarlyStopping(patience=5),
                        # StochasticWeightAveraging(swa_utils, swa_start=1, verbose=1, swa_lr=lr),
                        "accuracy", "balanced_accuracy","f1",("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
                        WandbLogger(wandb_run),
                        checkpoint,
                        ],  #"accuracy",
                    device='cuda')
    


    clf.initialize() # This is important!
    print('classifier initialized')

    # # load pre-trained model ##
    # if pre_trained:
    #     clf.initialize() # This is important!
    #     clf.load_params(
    #         f_params=load_path +'model.pkl', f_optimizer= load_path +'opt.pkl', f_history=load_path +'history.json')
    #     print('pre-trained model loaded')
    # # end of load pre-trained model ##

    ####### EVAL pre ############
    print('Evaluating before training')
    # if train_folder2:
    b_acc_merge, b_acc_tuh, b_acc_nmt, loss_merge, loss_tuh, loss_nmt = print_single_ds_performance(clf, window_test_set)
    save_as_csv(clf, seed, model_name,0, 0, b_acc_merge, b_acc_tuh, b_acc_nmt,write =False)

    if only_eval:
        # b_acc_merge, b_acc_tuh, b_acc_nmt, loss_merge, loss_tuh, loss_nmt =
        acc_dict = print_single_ds_performance_groups(clf, window_test_set)
        wandb_run.log(acc_dict)
        return b_acc_merge, b_acc_tuh, b_acc_nmt, loss_merge, loss_tuh, loss_nmt

        # save_as_csv(clf, seed, model_name,0, 0, b_acc_merge, b_acc_tuh, b_acc_nmt,write =False)
    # wandb.run.summary["loss_merge"] = loss_merge
    # wandb.run.summary["loss_tuh"] = loss_tuh
    # wandb.run.summary["loss_nmt"] = loss_nmt
    # if ids_to_load_train2 < 25:
    #     print("ids_to_load_train < 25 and returing results before training")
    #     return b_acc_merge, b_acc_tuh, b_acc_nmt, loss_merge, loss_tuh, loss_nmt


    ### end of eval ###

    print('AdamW')
    print("Number of parameters = ", sum(p.numel() for p in model.parameters() if p.requires_grad))



    # Model training for a specified number of epochs. `y` is None as it is already supplied
    if ds_all3:
        clf.fit(window_train_set_tueg, y=None, epochs=n_epochs)
    else:
        clf.fit(window_train_set, y=None, epochs=n_epochs)

    print('end training')

    ## load the best model
    clf.initialize()
    clf.load_params(checkpoint=checkpoint)
    print('best model loaded')
    ## end of load the best model 

    ### end of eval ###
    ####### EVAL ############
    # if train_folder2:

    if ds_all3:
        print('Evaluating on TUAB and NMT train sets, after training on TUEG')
        b_acc_merge, b_acc_tuh, b_acc_nmt, loss_merge, loss_tuh, loss_nmt = print_single_ds_performance(clf, window_test_set)
        save_as_csv(clf, seed, model_name,ids_to_load_train, ids_to_load_train2, b_acc_merge, b_acc_tuh, b_acc_nmt)
    elif ds_all4:
        print('Evaluating after training on Beyond  datasets')
        b_acc_merge, b_acc_tuh, b_acc_nmt, loss_merge, loss_tuh, loss_nmt = print_single_ds_performance_beyond(clf, windows_dataset_Beyond)
        save_as_csv(clf, seed, model_name,ids_to_load_train, ids_to_load_train2, b_acc_merge, b_acc_tuh, b_acc_nmt)
        print('Evaluating after training on TUAB and NMT train sets')
        b_acc_merge, b_acc_tuh, b_acc_nmt, loss_merge, loss_tuh, loss_nmt = print_single_ds_performance(clf, window_test_set)
        save_as_csv(clf, seed, model_name,ids_to_load_train, ids_to_load_train2, b_acc_merge, b_acc_tuh, b_acc_nmt)
    else:
        print('Evaluating after training on TUAB and NMT test sets')
        b_acc_merge, b_acc_tuh, b_acc_nmt, loss_merge, loss_tuh, loss_nmt = print_single_ds_performance(clf, window_test_set)
        save_as_csv(clf, seed, model_name,ids_to_load_train, ids_to_load_train2, b_acc_merge, b_acc_tuh, b_acc_nmt)

    ### end of eval ###

     ####### SAVE ############
    print('saving')
    # save model

    # Extract loss and accuracy values for plotting from history object
    plt.style.use('seaborn')
            
            #plots only loss
            
    results_columns = ['train_loss','valid_loss','train_accuracy','valid_accuracy']#, 'train_balanced_accuracy', 'valid_balanced_accuracy']

    # df = pd.DataFrame(clf.history[:, results_columns], columns=results_columns,index=clf.history[:, 'epoch'])
    # df.to_pickle(result_path + '_df_history.pkl')
    #save history
    torch.save(clf.history, result_path + '_clf_history.py')
    
    path = result_path + "model_{}.pt".format(seed)
    torch.save(clf.module, path)
    path = result_path + "state_dict_{}.pt".format(seed)
    torch.save(clf.module.state_dict(), path)

    clf.save_params(f_params=result_path +'model.pkl', f_optimizer= result_path +'opt.pkl', f_history=result_path +'history.json')

    
    pred_win = clf.predict_with_window_inds_and_ys(window_test_set)



    preds_per_trial= trial_preds_from_window_preds(pred_win['preds'], pred_win['i_window_in_trials'], pred_win['i_window_stops'])
    mean_preds_per_trial = [np.mean(preds, axis=1) for preds in
                                    preds_per_trial]
    mean_preds_per_trial = np.array(mean_preds_per_trial)
    y = window_test_set.description['pathological']
    column0, column1 = "non-pathological", "pathological"
    a_dict = {column0: mean_preds_per_trial[:, 0],
              column1: mean_preds_per_trial[:, 1],
              "true_pathological": y}

    assert len(y) == len(mean_preds_per_trial)

    # store predictions

    pd.DataFrame.from_dict(a_dict).to_csv(result_path + "predictions_eval_" + str(model_name) +
                                              ".csv")

    
    deep_preds =  mean_preds_per_trial[:, 0] <=  mean_preds_per_trial[:, 1]
    class_label = window_test_set.description['pathological']
    class_preds =deep_preds.astype(int)



    fig = plot_confusion_matrix_paper(confusion_matrix(class_label, class_preds),class_names=['normal', 'abnormal'])
    fig.savefig(result_path + 'confusion_matrix_eval.png')



    
    del model, clf, window_train_set
    return b_acc_merge, b_acc_tuh, b_acc_nmt, loss_merge, loss_tuh, loss_nmt



