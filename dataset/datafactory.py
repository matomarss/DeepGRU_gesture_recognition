from dataset.impl.leap_gestures import DatasetLeapGestures
from dataset.impl.sbu_kinect import DatasetSBUKinect


# ----------------------------------------------------------------------------------------------------------------------
class DataFactory:
    """
    A factory class for instantiating different datasets
    """
    dataset_names = [
            'sbu',
            'leap',
    ]

    @staticmethod
    def instantiate(dataset_name, num_synth, seq_len=1, pca=None, normalization=None, center_norm=False):
        """
        Instantiates a dataset with its name
        """

        if dataset_name not in DataFactory.dataset_names:
            raise Exception('Unknown dataset "{}"'.format(dataset_name))

        if dataset_name == "sbu":
            return DatasetSBUKinect(num_synth=num_synth)

        if dataset_name == 'leap':
            return DatasetLeapGestures(num_synth=num_synth,
                                       seq_len=seq_len, pca=pca, normalization=normalization, center_norm=center_norm)

        raise Exception('Unknown dataset "{}"'.format(dataset_name))
