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
    def instantiate(dataset_name, num_synth, center_norm=False, pca=None, lr=0.001, wd=0):
        """
        Instantiates a dataset with its name
        """

        if dataset_name not in DataFactory.dataset_names:
            raise Exception('Unknown dataset "{}"'.format(dataset_name))

        if dataset_name == "sbu":
            return DatasetSBUKinect(num_synth=num_synth)

        if dataset_name == 'leap':
            return DatasetLeapGestures(num_synth=num_synth, center_norm=center_norm, pca=pca, learning_rate=lr, weight_decay=wd)

        raise Exception('Unknown dataset "{}"'.format(dataset_name))
