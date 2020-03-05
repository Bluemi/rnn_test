import numpy as np

from data import chose_dataset_placeholder_from_database, Dataset
from util import show_frames


class RenderCrossSupplier:
    def __init__(self, annotations):
        """
        Renders the current frame and a little cross, for the annotations.

        :param annotations: The annotations to draw
        :type annotations: np.ndarray
        """
        self.annotations = annotations

    def __call__(self, frames_state, frames):
        """
        Renders the current frame and a little cross, for the annotations.

        :param frames_state: The current frame state
        :type frames_state: ShowFramesState
        :param frames: The frames to show
        :type frames: list[np.ndarray] or np.ndarray
        :return: The current frame
        :rtype: np.ndarray
        """
        index = frames_state.current_index
        current_frame = frames[index].copy()

        x, y = self.annotations[index].astype('int')

        current_frame[y][x] = np.array([255, 255, 255])

        return current_frame


def show_dataset(args):
    dataset_placeholder = chose_dataset_placeholder_from_database(args.database_directory)

    dataset = dataset_placeholder.load()

    render_callback = None
    if isinstance(dataset, Dataset):
        render_callback = RenderCrossSupplier(dataset.annotation_data)

    show_frames(dataset.video_data, frame_callback=render_callback)
