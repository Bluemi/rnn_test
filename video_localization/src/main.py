import argparse
import os
import time
import numpy as np

from camera import Camera
from util import FPS, RenderWindow, ESCAPE_KEY, EditFramesControl, show_frames

DATASET_TIME_FORMAT = '%H_%M_%S__%d_%m_%Y'


def parse_args():
    parser = argparse.ArgumentParser(description='Creates a dataset or tests the camera.')
    sub_parsers = parser.add_subparsers()

    # test camera
    test_camera_parser = sub_parsers.add_parser('test-camera', description='Test the connected camera.')
    test_camera_parser.set_defaults(func=test_camera)

    # create dataset
    create_dataset_parser = sub_parsers.add_parser(
        'create-dataset', description='Creates a dataset inside the given database.'
    )
    create_dataset_parser.add_argument(
        'database_directory', metavar='database-directory', type=str, help='The location for the database.'
    )
    create_dataset_parser.set_defaults(func=create_dataset)

    # show dataset
    show_dataset_parser = sub_parsers.add_parser('show-dataset', description='Shows the content of a dataset')
    show_dataset_parser.add_argument(
        'database_directory', metavar='database-directory', type=str, help='The location of the database'
    )
    show_dataset_parser.set_defaults(func=show_dataset)

    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


def test_camera(_args):
    """
    Opens a window and shows the video stream of the connected camera while printing fps information.

    :param _args: unused
    """
    del _args

    camera = Camera.create()
    render_window = RenderWindow('preview')

    fps = FPS()

    while True:
        frame = camera.next_frame()

        key = render_window.show_frame(frame, wait_key_duration=10)
        if key == ESCAPE_KEY:
            break

        fps.update()
        print(fps.get_fps(), flush=True)

    render_window.close()
    camera.close()


def create_dataset(args):
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


def show_dataset(args):
    dataset = chose_dataset(args.database_directory)

    show_frames(dataset)


if __name__ == '__main__':
    main()
