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
    def __init__(self, root="C:/Users/matom/OneDrive/Počítač/skola3/gestures_recognition/gestures/prepped", num_synth=0):
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
                                 num_epochs=40)

    def _get_augmenters(self, random_seed):
        return [
            AugRandomScale(3, self.num_synth, random_seed, 0.7, 1.3),
            AugRandomTranslation(3, self.num_synth, random_seed, -1, 1),
        ]

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
                label = npz_filename.split('_')[0]
                # if not once:
                #     print(features)
                for single_sequence_features in np.array_split(features, 5):
                    # if not once:
                    #     print(single_sequence_features)
                    sample = Sample(single_sequence_features, label, participant_name, path)
                    samples.append(sample)
                    s_idx = len(samples) - 1

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
            assert len(train_indices[fold_idx]) + len(test_indices[fold_idx]) == len(samples)
            # Ensure there is no intersection between training/test indices
            assert len(set(train_indices[fold_idx]).intersection(test_indices[fold_idx])) == 0

        return LowLevelDataset(samples, train_indices, test_indices)