import math
import os
from pathlib import Path
import numpy as np

from data.sbu.download_sbu import download_sbu
from dataset.dataset import Dataset, HyperParameterSet
from dataset.augmentation import AugRandomScale, AugRandomTranslation
from dataset.impl.lowlevel import Sample, LowLevelDataset
from utils.logger import log


# ----------------------------------------------------------------------------------------------------------------------
class DatasetLeapGestures(Dataset):
    def __init__(self, root="C:/Users/matom/OneDrive/Počítač/skola3/gestures_recognition/gestures/prepped", num_synth=0,
                 center_norm=False):
        self.sparse_division_num_parts = 5
        self.center_norm = center_norm
        super(DatasetLeapGestures, self).__init__("LeapGestures", root, num_synth)

    def _load_underlying_dataset(self):
        self.underlying_dataset = self._load_leap_gestures_dataset()
        self.num_features = 18  # 3D world coordinates joints of two people (15 joints x 3 dimensions).
                                # Each row is one person. Every two rows make up one frame.
        self.num_folds = 5      # This dataset has 5 folds

    def get_hyperparameter_set(self):
        return HyperParameterSet(learning_rate=0.001,
                                 batch_size=64,
                                 weight_decay=0,
                                 num_epochs=15)

    def _get_augmenters(self, random_seed):
        return [
            AugRandomScale(3, self.num_synth, random_seed, 0.7, 1.3),
            AugRandomTranslation(3, self.num_synth, random_seed, -1, 1),
        ]

    def apply_preprocessing(self, features):
        if self.center_norm is True:
            new_features = []
            for i in range(len(features)):
                palm_co_x = features[i][0]
                palm_co_y = features[i][1]
                palm_co_z = features[i][2]
                middle_co_x = features[i][9] - palm_co_x
                middle_co_y = features[i][10] - palm_co_y
                middle_co_z = features[i][11] - palm_co_z
                normalizer = math.sqrt(middle_co_x ** 2 + middle_co_y ** 2 + middle_co_z ** 2)
                new_features.append([features[i][0] / 200, features[i][1] / 200, features[i][2] / 200])
                for j in range(3, len(features[i]), 3):
                    new_features[i].append((features[i][j] - palm_co_x) / normalizer)
                    new_features[i].append((features[i][j + 1] - palm_co_y) / normalizer)
                    new_features[i].append((features[i][j + 2] - palm_co_z) / normalizer)
            new_features = np.array(new_features)
            return new_features
        else:
            return features

    def create_seq_data(self, features):
        features = np.array_split(features, self.sparse_division_num_parts)

        return features

    def _load_leap_gestures_dataset(self, unnormalize=True, verbose=False):
        """
        Loads the SBU Kinect Interactions dataset. We unnormalize the raw data using the equations
        that are provided in the dataset's documentation.
        """

        # Pre-set 5-fold cross validations from dataset's README
        # FOLD[i] means train on every other fold, test on fold i
        TESTING_SET=["palo", "zuzka", "stefan"]
        FOLDS = [
            ["jano", "janci"],
            ["viktor", "clara"],
            ["igor", "iveta"],
            ["barbora"],
            ["zdenka"]
        ]

        # Number of folds
        FOLD_CNT = len(FOLDS)

        # Number of joints
        JOINT_CNT = 15

        # Using 5-fold cross validation as the predefined test
        # (e.g. train[0] test[0] mean test on FOLD[0], train on everything else)
        train_indices = [[] for i in range(FOLD_CNT)]
        test_indices = [[] for i in range(FOLD_CNT)]
        final_test_indices = []
        samples = []

        participant_dir = [f for f in os.listdir(self.root)]
        # once = False
        for participant in participant_dir:
            participant_name = participant.split('_')[1]
            npz_filenames = os.listdir(os.path.join(self.root, participant))
            for npz_filename in npz_filenames:
                path = os.path.join(self.root, participant, npz_filename)
                data = np.load(path)

                features = data['features']
                features = self.apply_preprocessing(features)
                label = npz_filename.split('_')[0]
                # if not once:
                #     print(features)
                for sequence_features in self.create_seq_data(features):
                    # if not once:
                    # print(sequence_features)
                    sample = Sample(sequence_features, label, participant_name, path)
                    samples.append(sample)
                    s_idx = len(samples) - 1

                    if participant_name in TESTING_SET:
                        final_test_indices.append(s_idx)

                    for fold_idx in range(FOLD_CNT):
                        fold = FOLDS[fold_idx]

                        if participant_name in fold:
                            # Add the instance as a TESTING instance to this fold
                            test_indices[fold_idx].append(s_idx)

                            # For all other folds, this guy would be a TRAINING instance
                            for other_idx in range(FOLD_CNT):
                                if fold_idx == other_idx:
                                    continue

                                train_indices[other_idx].append(s_idx)
                # once = True

        # k-fold sanity check
        for fold_idx in range(FOLD_CNT):
            assert len(train_indices[fold_idx]) + len(test_indices[fold_idx]) + len(final_test_indices) == len(samples)
            # Ensure there is no intersection between training/test/final test indices
            assert len(set(train_indices[fold_idx]).intersection(test_indices[fold_idx])) == 0
            assert len(set(train_indices[fold_idx]).intersection(final_test_indices)) == 0
            assert len(set(final_test_indices).intersection(test_indices[fold_idx])) == 0

        return LowLevelDataset(samples, train_indices, test_indices)