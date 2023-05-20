import numpy as np

import torch

from dataset.datafactory import DataFactory
from main import run_fold
from utils.logger import log                  # Logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# ----------------------------------------------------------------------------------------------------------------------
seed = 1570254494
use_cuda = torch.cuda.is_available()
dataset_name = "leap"


# ----------------------------------------------------------------------------------------------------------------------
def log_and_train(stage, scaler, center_norm, pca):
    log('-----Stage {}-----'.format(stage))
    log('Evaluated preprocessing: (center-norm: "{}", scaler: "{}", pca: "{}")'
        .format(center_norm, scaler, pca))

    avg_accuracy = train(scaler=scaler, center_norm=center_norm, pca=pca)

    log('')
    log('-----------------------------------------------------------------------')
    log('Training for stage {} complete!'.format(stage))
    log('Evaluated preprocessing is: (center-norm: "{}", scaler: "{}", pca: "{}")'
        .format(center_norm, scaler, pca))
    log('Average accuracy is: {}'.format(avg_accuracy))
    log('')
    log('')
    log('-----------------------------------------------------------------------')
    log('-----------------------------------------------------------------------')


# ----------------------------------------------------------------------------------------------------------------------
def main():
    """
        Conduct experiments on combinations of preprocessing using GridSearch method
    """
    log.set_dataset_name(dataset_name)

    log('----------------Training initialized----------------')
    stage = 1
    for pca in [PCA(n_components=18), PCA(n_components=12), PCA(n_components=11), PCA(n_components=8), PCA(n_components=5), PCA(n_components=4), PCA(n_components=2)]:
        for scaler in [StandardScaler(), MinMaxScaler()]:
            for center_norm in [False, True]:
                log_and_train(stage=stage, scaler=scaler, center_norm=center_norm, pca=pca)
                stage += 1
    pca = None
    for scaler in [None, StandardScaler(), MinMaxScaler()]:
        for center_norm in [False, True]:
            log_and_train(stage=stage, scaler=scaler, center_norm=center_norm, pca=pca)
            stage += 1


# ----------------------------------------------------------------------------------------------------------------------
def train(num_synth=0, center_norm=False, scaler=None, pca=None):
    # Load the dataset
    dataset = DataFactory.instantiate(dataset_name, num_synth, center_norm=center_norm, pca=pca)
    log.log_dataset(dataset)
    log("Random seed: " + str(seed))
    torch.manual_seed(seed)

    # Run each fold and average the results
    accuracies = []

    for fold_idx in range(dataset.num_folds):
        log('Running fold "{}"...'.format(fold_idx))

        test_accuracy = run_fold(dataset, fold_idx, use_cuda, scaler)
        accuracies += [test_accuracy]

        log('Fold "{}" complete, final accuracy: {}'.format(fold_idx, test_accuracy))

    return np.mean(accuracies)


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
