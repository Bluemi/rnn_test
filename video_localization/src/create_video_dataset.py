import os
import time

import numpy as np

from camera import Camera
from util import RenderWindow, ESCAPE_KEY, EditFramesControl, show_frames

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
        if key == ESCAPE_KEY:
            break

    render_window.close()
    camera.close()

    chosen_frames = choose_frames(frames)

    result = np.array(chosen_frames)

    file_path = os.path.join(dataset_directory, 'data.npy')
    with open(file_path, 'wb') as f:
        # noinspection PyTypeChecker
        np.save(f, result)


def choose_frames(frames):
    """
    Manipulates the given frames.

    :param frames: A list of ndarray containing the recorded videos
    :type frames: list[np.ndarray]
    """
    edit_control = EditFramesControl(len(frames))

    show_frames(frames, edit_control, 'choose frames')

    return frames[edit_control.start_index:edit_control.end_index]
