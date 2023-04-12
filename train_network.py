import numpy as np

import torch

from dataset.datafactory import DataFactory
from main import run_fold
from utils.logger import log                  # Logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ----------------------------------------------------------------------------------------------------------------------
seed = 1570254494
use_cuda = torch.cuda.is_available()
dataset_name = "leap"


# ----------------------------------------------------------------------------------------------------------------------
def main():
    log.set_dataset_name(dataset_name)

    log('----------------Training initialized----------------')
    stage = 1
    for seq_len in [20, 10, 1, 40]:
        # for pca in [PCA(n_components=12), None, PCA(n_components=2), PCA(n_components=4), PCA(n_components=5), PCA(n_components=8),
        #             PCA(n_components=11)]:
        for normalization in [None, StandardScaler(), MinMaxScaler()]:
            log('-----Stage {}-----'.format(stage))
            log('Evaluated frame sequence length: {}'.format(seq_len))
            log('Evaluated preprocessing: (normalization: "{}", pca: "{}")'.format(normalization, None))

            avg_accuracy = train(seq_len=seq_len, pca=None, normalization=normalization)

            log('')
            log('-----------------------------------------------------------------------')
            log('Training for stage {} complete!'.format(stage))
            log('Evaluated frame sequence length was: {}'.format(seq_len))
            log('Evaluated preprocessing was: (normalization: "{}", pca: "{}")'.format(normalization, None))
            log('Average accuracy is: {}'.format(avg_accuracy))
            log('')
            log('')
            log('-----------------------------------------------------------------------')
            log('-----------------------------------------------------------------------')

            stage += 1


# ----------------------------------------------------------------------------------------------------------------------
def train(num_synth=0, seq_len=1, pca=None, normalization=None, center_norm=False):
    # Load the dataset
    dataset = DataFactory.instantiate(dataset_name, num_synth,
                                      seq_len=seq_len, pca=pca, normalization=normalization, center_norm=center_norm)
    log.log_dataset(dataset)
    log("Random seed: " + str(seed))
    torch.manual_seed(seed)

    # Run each fold and average the results
    accuracies = []

    for fold_idx in range(dataset.num_folds):
        log('Running fold "{}"...'.format(fold_idx))

        test_accuracy = run_fold(dataset, fold_idx, use_cuda)
        accuracies += [test_accuracy]

        log('Fold "{}" complete, final accuracy: {}'.format(fold_idx, test_accuracy))

    return np.mean(accuracies)


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
