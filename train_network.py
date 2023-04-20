import numpy as np

import torch

from enum import Enum
from dataset.datafactory import DataFactory
from main import run_fold
from utils.logger import log                  # Logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ----------------------------------------------------------------------------------------------------------------------
seed = 1570254494
use_cuda = torch.cuda.is_available()
dataset_name = "leap"


class Norm(Enum):
    STD = 1
    MIN_MAX = 2
    NONE = None


class Seq(Enum):
    SPARSE = 1
    DENSE = 2

# ----------------------------------------------------------------------------------------------------------------------
def main():
    log.set_dataset_name(dataset_name)

    log('----------------Training initialized----------------')
    stage = 1
    for seq in [Seq.SPARSE, Seq.DENSE]:
        for pca in [PCA(n_components=12), None,
                    # PCA(n_components=2), PCA(n_components=4), PCA(n_components=5), PCA(n_components=8),
                    #  PCA(n_components=11)
        ]:
            for normalization in [Norm.STD, Norm.NONE]:
                log('-----Stage {}-----'.format(stage))
                log('Evaluated sequentization: {}'.format(seq.name))
                log('Evaluated preprocessing: (normalization: "{}", pca: "{}")'.format(normalization.name, pca))

                avg_accuracy = train(pca=pca, normalization=normalization, seq=seq)

                log('')
                log('-----------------------------------------------------------------------')
                log('Training for stage {} complete!'.format(stage))
                log('Evaluated sequentization was: {}'.format(seq.name))
                log('Evaluated preprocessing is: (normalization: "{}", pca: "{}")'.format(normalization.name, pca))
                log('Average accuracy is: {}'.format(avg_accuracy))
                log('')
                log('')
                log('-----------------------------------------------------------------------')
                log('-----------------------------------------------------------------------')

                stage += 1


# ----------------------------------------------------------------------------------------------------------------------
def train(num_synth=0, pca=None, center_norm=False, normalization=Norm.NONE, seq=Seq.SPARSE):
    if normalization == Norm.STD:
        normalize = True
    else:
        normalize = False
    # Load the dataset
    dataset = DataFactory.instantiate(dataset_name, num_synth,
                                      pca=pca, center_norm=center_norm, seq=seq)
    log.log_dataset(dataset)
    log("Random seed: " + str(seed))
    torch.manual_seed(seed)

    # Run each fold and average the results
    accuracies = []

    for fold_idx in range(dataset.num_folds):
        log('Running fold "{}"...'.format(fold_idx))

        test_accuracy = run_fold(dataset, fold_idx, use_cuda, normalize)
        accuracies += [test_accuracy]

        log('Fold "{}" complete, final accuracy: {}'.format(fold_idx, test_accuracy))

    return np.mean(accuracies)


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
