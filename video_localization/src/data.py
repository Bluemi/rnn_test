import os

import numpy as np


class VideoDataset:
    def __init__(self, data):
        """
        Creates a new VideoDataset.
        A VideoDataset only contains the video information, but no hand position information.

        :param data: The video data
        :type data: np.ndarray
        """
        self.data = data


class HandPositionDataset:
    def __init__(self, data):
        """
        Creates a new HandPositionDataset.
        A HandPositionDataset only contains the hand position information, but no video information.

        :param data: The hand position data
        :type data: np.ndarray
        """
        self.data = data


class Dataset:
    def __init__(self, video_data, hand_position_data):
        """
        Contains the video data as well as the hand position data

        :param video_data: The video data
        :type video_data: np.ndarray
        :param hand_position_data:
        :type hand_position_data: np.ndarray
        """
        self.video_data = video_data
        self.hand_position_data = hand_position_data

        assert(video_data.shape[0] == hand_position_data.shape[0])

    def get_num_samples(self):
        return self.video_data.shape[0]

    def get_resolution(self):
        return self.video_data.shape[1:]


def chose_dataset(database_dir):
    dataset_names = os.listdir(database_dir)

    for index, dataset_name in enumerate(dataset_names):
        print('{}: {}'.format(index, dataset_name))

    dataset_index = None
    while dataset_index is None:
        user_input = input('\nchoose the dataset> ')
        try:
            dataset_index = int(user_input)
        except ValueError:
            print('Use the index.')
            continue
        if dataset_index < 0 or dataset_index >= len(dataset_names):
            print('Index out of range.')
            dataset_index = None
            continue

    dataset_path = os.path.join(database_dir, dataset_names[dataset_index], 'data.npy')
    return np.load(dataset_path)
