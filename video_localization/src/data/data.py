import copy
import json
import os
from enum import Enum
from typing import Iterable

import numpy as np

from util.util import always_true

VIDEO_DATASET_FILENAME = 'video_data.npy'
ANNOTATION_DATASET_FILENAME = 'annotations.npy'
INFO_FILENAME = 'info.json'
MAX_PIXEL_VALUE = 255


class VideoDataset:
    def __init__(self, name, data_info, video_data):
        """
        Creates a new VideoDataset.
        A VideoDataset only contains the video information, but no hand position information.

        :param name: The name of the dataset
        :type name: str
        :param data_info: The info about this dataset
        :type data_info: DataInfo
        :param video_data: The video data
        :type video_data: np.ndarray
        """
        self.name = name
        self.data_info = data_info
        self.video_data = video_data

    def get_num_samples(self):
        return self.video_data.shape[0]

    def get_resolution(self):
        return self.video_data.shape[1:]

    def get_num_bytes(self):
        return self.video_data.nbytes

    @staticmethod
    def from_placeholder(placeholder):
        """
        Loads a new VideoDataset from the given placeholder.

        :param placeholder: The DatasetPlaceholder to load
        :type placeholder: DatasetPlaceholder
        :return: A new VideoDataset
        :rtype: VideoDataset
        """
        video_data = np.load(placeholder.get_video_path()).astype(np.float32) / MAX_PIXEL_VALUE

        return VideoDataset(placeholder.get_basename(), data_info=placeholder.data_info, video_data=video_data)

    def is_full_dataset(self):
        return False

    def is_video_dataset(self):
        return True


class AnnotatedDataset(VideoDataset):
    def __init__(self, name, data_info, video_data, annotation_data):
        """
        Contains the video data as well as the hand position data

        :param name: The name of the dataset
        :type name: str
        :param data_info: The data info about this dataset
        :type data_info: DataInfo
        :param video_data: The video data
        :type video_data: np.ndarray
        :param annotation_data:
        :type annotation_data: np.ndarray
        """
        super().__init__(name, data_info, video_data)
        self.annotation_data = annotation_data

        if not video_data.shape[0] == annotation_data.shape[0] == data_info.num_samples:
            raise DataError(
                'num samples not matching: video data={} annotation_data={} data_info={}'
                .format(video_data.shape[0], annotation_data.shape[0], data_info.num_samples)
            )

    def __str__(self):
        return self.name

    def __repr__(self):
        return 'Dataset<{}; {} samples>'.format(self.name, self.data_info.num_samples)

    def get_num_bytes(self):
        return self.video_data.nbytes + self.annotation_data.nbytes

    @staticmethod
    def from_placeholder(placeholder, divisible_by=None):
        """
        Loads a new Dataset from the given placeholder.

        :param placeholder: The DatasetPlaceholder to load
        :type placeholder: DatasetPlaceholder
        :param divisible_by: If set, the dataset will contain a number of samples that is divisible by the given number
        :type divisible_by: int
        :return: A new Dataset
        :rtype: AnnotatedDataset
        """
        data_info = copy.deepcopy(placeholder.data_info)
        if placeholder.dataset_type != DatasetPlaceholder.DatasetType.ANNOTATED_DATASET:
            raise DataError('Cant load annotated Dataset from video dataset placeholder')

        # datasets are saved as int array with values between 0 and 255. Convert it to float.
        video_data = np.load(placeholder.get_video_path()).astype(np.float32) / MAX_PIXEL_VALUE
        annotations_data = np.load(placeholder.get_annotations_path())

        if divisible_by is not None:
            num_samples = (data_info.num_samples // divisible_by) * divisible_by
            data_info.num_samples = num_samples
            video_data = video_data[:num_samples]
            annotations_data = annotations_data[:num_samples]

        return AnnotatedDataset(placeholder.get_basename(), data_info, video_data, annotations_data)

    @staticmethod
    def concatenate(datasets):
        """
        Returns the given datasets concatenated into one np array.

        :param datasets: The datasets to concatenate
        :type datasets: Iterable[AnnotatedDataset]
        :return: A new Dataset that contains all given datasets
        :rtype: AnnotatedDataset
        """
        datasets = list(datasets)
        video_data = list(map(lambda dataset: dataset.video_data, datasets))
        annotations = list(map(lambda dataset: dataset.annotation_data, datasets))

        concatenated_video_data = np.concatenate(video_data)
        concatenated_annotation_data = np.concatenate(annotations)

        joined_info = DataInfo.join(map(lambda dataset: dataset.data_info, datasets))

        return AnnotatedDataset('train_dataset', joined_info, concatenated_video_data, concatenated_annotation_data)

    def is_full_dataset(self):
        return True

    def is_video_dataset(self):
        return False

    @staticmethod
    def load_database(database_directory):
        """
        Loads all annotated datasets from the given base directory.

        :param database_directory: The database directory to load
        :type database_directory: str
        :return: A new Dataset loaded from the given directory
        :rtype: AnnotatedDataset
        """
        dataset_placeholders = DatasetPlaceholder.list_database(
            database_directory, dataset_filter=DatasetPlaceholder.is_full_dataset
        )
        datasets = list(map(AnnotatedDataset.from_placeholder, dataset_placeholders))
        return AnnotatedDataset.concatenate(datasets)


class DataInfo:
    def __init__(self, resolution, num_samples, subjects, tags):
        """
        Creates a DataInfo object.

        :param resolution: The resolution of the Dataset. Given as tuple (height, width, depth). The parameter depth can
                           be omitted.
        :type resolution: tuple[int, int, int] or tuple[int, int]
        :param num_samples: The number of samples in this dataset
        :type num_samples: int
        :param subjects: A list of the subjects seen in this dataset
        :type subjects: list[str]
        :param tags: List of tags for this dataset
        :type tags: list[str]
        """
        self.resolution = resolution
        self.num_samples = num_samples
        self.subjects = subjects
        self.tags = tags

    @staticmethod
    def from_info_file(filepath):
        """
        Reads the dataset info from the given path.

        :param filepath: The info file to read
        :type filepath: str
        :return: An DataInfo object containing the information of the file
        :rtype: DataInfo
        """
        with open(filepath, 'r') as f:
            info = json.load(f)
            return DataInfo(info['resolution'], info['num_samples'], info['subjects'], info['tags'])

    @staticmethod
    def join(data_infos):
        """
        Joins the given data info objects.

        :param data_infos: A list of DataInfo objects to join
        :type data_infos: Iterable[DataInfo]
        :return: The joined DataInfo
        :rtype: DataInfo
        """
        resolution = None
        num_samples = 0
        subjects = set()
        tags = set()
        for data_info in data_infos:
            if resolution is None:
                resolution = data_info.resolution
            else:
                assert resolution == data_info.resolution, 'Cannot join data infos with different resolutions'

            num_samples += data_info.num_samples

            for subject in data_info.subjects:
                subjects.add(subject)

            for tag in data_info.tags:
                tags.add(tag)

        return DataInfo(resolution, num_samples, list(subjects), list(tags))


class DatasetPlaceholder:
    class DatasetType(Enum):
        VIDEO_DATASET = 0
        ANNOTATED_DATASET = 1

    def __init__(self, path, dataset_type, data_info):
        """
        Creates a new DatasetPlaceholder.

        :param path: The path of the dataset
        :type path: str
        :param dataset_type: The type of the dataset. Either VIDEO_DATASET or ANNOTATED_DATASET
        :type dataset_type: DatasetPlaceholder.DatasetType
        :param data_info: The info object for this dataset
        :type data_info: DataInfo
        """
        self.path = path
        self.dataset_type = dataset_type
        self.data_info = data_info

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

    def get_info_path(self):
        """
        Returns the path to the info data.

        :return: the path to the info data
        :rtype: str
        """
        return os.path.join(self.path, INFO_FILENAME)

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
        return self.dataset_type == DatasetPlaceholder.DatasetType.ANNOTATED_DATASET

    def load(self):
        """
        Returns a Dataset of VideoDataset depending on dataset_type

        :return: a Dataset of VideoDataset depending on dataset_type
        :rtype: AnnotatedDataset or VideoDataset
        """
        if self.dataset_type == DatasetPlaceholder.DatasetType.ANNOTATED_DATASET:
            return AnnotatedDataset.from_placeholder(self)
        elif self.dataset_type == DatasetPlaceholder.DatasetType.VIDEO_DATASET:
            return VideoDataset.from_placeholder(self)

        raise DataError('Unknown dataset type: {}'.format(self.dataset_type))

    def __str__(self):
        return '{} ({}; {} samples, subjects: [{}], tags: [{}])'.format(
            self.get_basename(),
            self.dataset_type.name.lower(),
            self.data_info.num_samples,
            ','.join(self.data_info.subjects),
            ','.join(self.data_info.tags)
        )

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
        # annotations
        annotations_filename = os.path.join(path, ANNOTATION_DATASET_FILENAME)
        if os.path.isfile(annotations_filename):
            dataset_type = DatasetPlaceholder.DatasetType.ANNOTATED_DATASET
        else:
            dataset_type = DatasetPlaceholder.DatasetType.VIDEO_DATASET

        # video
        video_data_filename = os.path.join(path, VIDEO_DATASET_FILENAME)
        if not os.path.isfile(video_data_filename):
            raise DataError('Could not find video data file: {}'.format(video_data_filename))

        # info
        info_data_filename = os.path.join(path, INFO_FILENAME)
        if not os.path.isfile(info_data_filename):
            raise DataError('Could not find info data file: {}'.format(info_data_filename))
        data_info = DataInfo.from_info_file(info_data_filename)

        return DatasetPlaceholder(path, dataset_type, data_info)

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
