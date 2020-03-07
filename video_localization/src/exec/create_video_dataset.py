import os
import time

import numpy as np

from util.camera import Camera
from data.data import VIDEO_DATASET_FILENAME
from util.util import RenderWindow, show_frames, default_key_callback, KeyCodes

DATASET_TIME_FORMAT = '%H_%M_%S__%d_%m_%Y'


def create_video_dataset(args):
    """
    Creates a new image dataset.

    :param args: A Namespace object containing arguments for the dataset creation.
                 See parse_args() for more information.
    :type args: argparse.Namespace

    :raise OSError: If the dataset directory already exists
    """
    dataset_directory = os.path.join(args.database_directory, 'dataset_{}'.format(time.strftime(DATASET_TIME_FORMAT)))

    os.makedirs(dataset_directory)

    camera = Camera.create()
    render_window = RenderWindow('current')

    frames = []

    while True:
        frame = camera.next_frame()

        frames.append(frame)

        key = render_window.show_frame(frame, wait_key_duration=10)
        if key == KeyCodes.ESCAPE_KEY:
            break

    render_window.close()
    camera.close()

    chosen_frames = choose_frames(frames)

    result = np.array(chosen_frames)

    file_path = os.path.join(dataset_directory, VIDEO_DATASET_FILENAME)
    with open(file_path, 'wb') as f:
        # noinspection PyTypeChecker
        np.save(f, result)


class EditKeySupplier:
    def __init__(self, num_frames):
        self.start_index = 0
        self.end_index = num_frames

    def __call__(self, frames_state, key):
        """
        Changes the frames_state, depending on key.

        :param frames_state: The ShowFramesState object to handle
        :type frames_state: ShowFramesState
        :param key: The pressed key
        :type key: int
        :return: True, if the key was applied otherwise False
        :rtype: bool
        """
        if default_key_callback(frames_state, key):
            return True

        if key == KeyCodes.ENTER_KEY:
            frames_state.running = False
        if key == KeyCodes.A_KEY:
            self.start_index = frames_state.current_index
            print('set start index = {}'.format(self.start_index), flush=True)
        elif key == KeyCodes.E_KEY:
            self.end_index = frames_state.current_index + 1
            print('set end index = {}'.format(self.end_index))
        else:
            return False

        return True


def choose_frames(frames):
    """
    Manipulates the given frames.

    :param frames: A list of ndarray containing the recorded videos
    :type frames: list[np.ndarray]
    """
    edit_key_supplier = EditKeySupplier(len(frames))

    show_frames(frames, 'choose frames', key_callback=edit_key_supplier)

    return frames[edit_key_supplier.start_index:edit_key_supplier.end_index]
