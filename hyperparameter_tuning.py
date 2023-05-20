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
def main():
    log.set_dataset_name(dataset_name)

    log('----------------Training initialized----------------')
    stage = 1
    pca = PCA(n_components=12)
    scaler = StandardScaler()
    center_norm = False
    for lr in [0.001, 0.01, 0.0001, 0.0008, 0.0005, 0.002]:
        for wd in [0, 0.0001, 0.001]:
            log('-----Stage {}-----'.format(stage))
            log('Preprocessing: (center-norm: "{}", scaler: "{}", pca: "{}")'
                .format(center_norm, scaler, pca))
            log('Evaluated learning rate: "{}"'.format(lr))
            log('Evaluated weight decay: "{}"'.format(wd))

            avg_accuracy = train(scaler=scaler, center_norm=center_norm, pca=pca, lr=lr, wd=wd)

            log('')
            log('-----------------------------------------------------------------------')
            log('Training for stage {} complete!'.format(stage))
            log('Evaluated learning rate was: "{}"'.format(lr))
            log('Evaluated weight decay was: "{}"'.format(wd))
            log('Average accuracy is: {}'.format(avg_accuracy))
            log('')
            log('')
            log('-----------------------------------------------------------------------')
            log('-----------------------------------------------------------------------')

            stage += 1


# ----------------------------------------------------------------------------------------------------------------------
def train(lr, wd, num_synth=0, center_norm=False, scaler=None, pca=None):
    # Load the dataset
    dataset = DataFactory.instantiate(dataset_name, num_synth, center_norm=center_norm, pca=pca, lr=lr, wd=wd)
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
