import os
from enum import Enum

import numpy as np

from util.util import always_true

VIDEO_DATASET_FILENAME = 'video_data.npy'
ANNOTATION_DATASET_FILENAME = 'annotations.npy'
MAX_PIXEL_VALUE = 255


class VideoDataset:
    def __init__(self, name,  video_data):
        """
        Creates a new VideoDataset.
        A VideoDataset only contains the video information, but no hand position information.

        :param name: The name of the dataset
        :type name: str
        :param video_data: The video data
        :type video_data: np.ndarray
        """
        self.name = name
        self.video_data = video_data

    def get_num_samples(self):
        return self.video_data.shape[0]

    def get_resolution(self):
        return self.video_data.shape[1:]

    @staticmethod
    def from_placeholder(placeholder):
        """
        Loads a new VideoDataset from the given placeholder.

        :param placeholder: The DatasetPlaceholder to load
        :type placeholder: DatasetPlaceholder
        :return: A new VideoDataset
        :rtype: VideoDataset
        """
        video_data = np.load(placeholder.get_video_path()).astype(np.float) / MAX_PIXEL_VALUE

        return VideoDataset(placeholder.get_basename(), video_data)


class Dataset(VideoDataset):
    def __init__(self, name, video_data, annotation_data):
        """
        Contains the video data as well as the hand position data

        :param name: The name of the dataset
        :type name: str
        :param video_data: The video data
        :type video_data: np.ndarray
        :param annotation_data:
        :type annotation_data: np.ndarray
        """
        super().__init__(name, video_data)
        self.annotation_data = annotation_data

        assert (video_data.shape[0] == annotation_data.shape[0])

    def __str__(self):
        return self.name

    def __repr__(self):
        return 'Dataset<{}>'.format(self.name)

    @staticmethod
    def from_placeholder(placeholder):
        """
        Loads a new Dataset from the given placeholder.

        :param placeholder: The DatasetPlaceholder to load
        :type placeholder: DatasetPlaceholder
        :return: A new Dataset
        :rtype: Dataset
        """
        if placeholder.dataset_type != DatasetPlaceholder.DatasetType.FULL_DATASET:
            raise DataError('Cant load full Dataset from video dataset')

        # datasets are saved as int array with values between 0 and 255. Convert it to float.
        video_data = np.load(placeholder.get_video_path()).astype(np.float) / MAX_PIXEL_VALUE
        annotations_data = np.load(placeholder.get_annotations_path())

        return Dataset(placeholder.get_basename(), video_data, annotations_data)

    @staticmethod
    def concatenate(datasets):
        """
        Returns the given datasets concatenated into one np array.

        :param datasets: The datasets to concatenate
        :type datasets: list[Dataset]
        :return: A new Dataset that contains all given datasets
        :rtype: Dataset
        """
        video_data = list(map(lambda dataset: dataset.video_data, datasets))
        annotations = list(map(lambda dataset: dataset.annotation_data, datasets))

        concatenated_video_data = np.concatenate(video_data)
        concatenated_annotation_data = np.concatenate(annotations)

        return Dataset('train_dataset', concatenated_video_data, concatenated_annotation_data)


class DatasetPlaceholder:
    class DatasetType(Enum):
        VIDEO_DATASET = 0
        FULL_DATASET = 1

    def __init__(self, path, dataset_type):
        """
        Creates a new DatasetPlaceholder.

        :param path: The path of the dataset
        :type path: str
        :param dataset_type:
        """
        self.path = path
        self.dataset_type = dataset_type

    def get_basename(self):
        """
        Returns the basename of the dataset path.

        :return: the basename of the dataset path
        :rtype: str
        """
        return os.path.basename(self.path)

    def get_video_path(self):
        """
        Returns the path to the video data.

        :return: the path to the video data
        :rtype: str
        """
        return os.path.join(self.path, VIDEO_DATASET_FILENAME)

    def get_annotations_path(self):
        """
        Returns the path to the annotation data.

        :return: the path to the annotation data
        :rtype: str
        """
        return os.path.join(self.path, ANNOTATION_DATASET_FILENAME)

    def is_video_dataset(self):
        """
        Returns whether it is a video dataset or not.
        :return: True, if it is a video dataset, otherwise False
        :rtype: bool
        """
        return self.dataset_type == DatasetPlaceholder.DatasetType.VIDEO_DATASET

    def is_full_dataset(self):
        """
        Returns whether it is a full dataset or not.
        :return: True, if it is a full dataset, otherwise False
        :rtype: bool
        """
        return self.dataset_type == DatasetPlaceholder.DatasetType.FULL_DATASET

    def load(self):
        """
        Returns a Dataset of VideoDataset depending on dataset_type

        :return: a Dataset of VideoDataset depending on dataset_type
        :rtype: Dataset or VideoDataset
        """
        if self.dataset_type == DatasetPlaceholder.DatasetType.FULL_DATASET:
            return Dataset.from_placeholder(self)
        elif self.dataset_type == DatasetPlaceholder.DatasetType.VIDEO_DATASET:
            return VideoDataset.from_placeholder(self)

        raise DataError('Unknown dataset type: {}'.format(self.dataset_type))

    def __str__(self):
        return '{} ({})'.format(self.get_basename(), self.dataset_type.name.lower())

    def __repr__(self):
        return self.get_basename()

    @staticmethod
    def from_directory(path):
        """
        Creates a new DatasetPlaceholder from a given path.

        :param path: The path to inspect
        :type path: str
        :return: A new DatasetPlaceholder
        :rtype: DatasetPlaceholder

        :raise DataError: If no video datafile could be found
        """
        annotations_filename = os.path.join(path, ANNOTATION_DATASET_FILENAME)
        if os.path.isfile(annotations_filename):
            dataset_type = DatasetPlaceholder.DatasetType.FULL_DATASET
        else:
            dataset_type = DatasetPlaceholder.DatasetType.VIDEO_DATASET

        video_data_filename = os.path.join(path, VIDEO_DATASET_FILENAME)
        if not os.path.isfile(video_data_filename):
            raise DataError('Could not find video data file: {}'.format(video_data_filename))

        return DatasetPlaceholder(path, dataset_type)

    @staticmethod
    def list_database(database_path, dataset_filter=None):
        """
        Returns a list of DatasetPlaceholders from a given database path.

        :param database_path: The path of the database
        :type database_path: str
        :param dataset_filter: A callable that is used to filter the datasets.
                               Should expect a DatasetPlaceholder and return True or False
        :type dataset_filter: Callable[[DatasetPlaceholder], bool] or None
        :return: A list of DatasetPlaceholders representing the database
        :rtype: list[DatasetPlaceholder]
        """
        if dataset_filter is None:
            dataset_filter = always_true

        if not os.path.isdir(database_path):
            raise DataError('Database {} does not exist'.format(database_path))

        sub_dirs = os.listdir(database_path)

        placeholders = []

        for subdir in sub_dirs:
            try:
                dataset_dir = os.path.join(database_path, subdir)
                placeholder = DatasetPlaceholder.from_directory(dataset_dir)
                if dataset_filter(placeholder):
                    placeholders.append(placeholder)
            except DataError:
                pass

        return placeholders


def chose_dataset_placeholder_from_database(database_dir):
    """
    Chooses a dataset placeholder from the given dataset placeholders.

    :param database_dir: The database to list datasets from
    :type database_dir: str
    :return: A DatasetPlaceholder
    :rtype: DatasetPlaceholder
    """
    dataset_placeholders = DatasetPlaceholder.list_database(database_dir)
    return chose_dataset_placeholder(dataset_placeholders)


def chose_dataset_placeholder(dataset_placeholders):
    """
    Chooses a dataset placeholder from the given dataset placeholders.

    :param dataset_placeholders: The placeholders to chose from
    :type dataset_placeholders: list[DatasetPlaceholder]
    :return: A DatasetPlaceholder
    :rtype: DatasetPlaceholder
    """
    for index, dataset_name in enumerate(dataset_placeholders):
        print('{}: {}'.format(index, dataset_name))

    dataset_index = None
    while dataset_index is None:
        user_input = input('\nchoose the dataset> ')
        try:
            dataset_index = int(user_input)
        except ValueError:
            print('Use the index.')
            continue
        if dataset_index < 0 or dataset_index >= len(dataset_placeholders):
            print('Index out of range.')
            dataset_index = None
            continue

    return dataset_placeholders[dataset_index]


class DataError(Exception):
    pass