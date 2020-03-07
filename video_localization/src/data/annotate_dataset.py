import numpy as np


from data.data import DatasetPlaceholder, chose_dataset_placeholder, VideoDataset, Dataset
from util.util import show_frames


def annotate_dataset(args):
    dataset_placeholders = DatasetPlaceholder.list_database(args.database_directory)
    video_dataset_placeholders = list(filter(DatasetPlaceholder.is_video_dataset, dataset_placeholders))

    dataset_placeholder = chose_dataset_placeholder(video_dataset_placeholders)

    source_dataset = dataset_placeholder.load()

    annotated_dataset = annotate_frames(source_dataset)

    np.save(dataset_placeholder.get_annotations_path(), annotated_dataset.annotation_data)


class EditMouseSupplier:
    def __init__(self, num_samples):
        """
        Creates a new EditMouseSupplier.

        :param num_samples: The number of samples
        :type num_samples: int
        """
        self.annotations = np.zeros((num_samples, 2), dtype=np.float) - 1
        self.x = None
        self.y = None

    def _is_set(self):
        return self.x is not None and self.y is not None

    def _set_point(self, frames_state):
        if self._is_set():
            self.annotations[frames_state.current_index][0] = self.x
            self.annotations[frames_state.current_index][1] = self.y
            print('set point', flush=True)
        else:
            print('Could not set point, because x, y was not defined.', flush=True)

    def is_point_set(self, index):
        return self.annotations[index][0] != -1 and self.annotations[index][1] != -1

    def __call__(self, event_type, x, y, _unused1, frames_state):
        """
        Sets the annotations, if mouse is pressed.

        :param event_type: The mouse event type. 0 -> Mouse Move, 1 -> Left Mouse Click
        :param x: The x position of the mouse
        :param y: The y position of the mouse
        :param frames_state: The current frames_state object
        """
        if event_type == 0:
            self.x = x
            self.y = y
        elif event_type == 1:
            self._set_point(frames_state)


class RenderCrossSupplier:
    def __init__(self, edit_mouse_supplier):
        """
        Renders the current frame and a little cross, for the annotations.

        :param edit_mouse_supplier: The EditMouseSupplier to get the annotations from
        :type edit_mouse_supplier: EditMouseSupplier
        """
        self.edit_mouse_supplier = edit_mouse_supplier

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
        if self.edit_mouse_supplier.is_point_set(index):
            current_frame = frames[index].copy()
            x, y = self.edit_mouse_supplier.annotations[index].astype('int')

            current_frame[y][x] = np.array([255, 255, 255])
        else:
            current_frame = frames[index]

        return current_frame


def annotate_frames(dataset):
    """
    Returns the full dataset.

    :param dataset: The dataset to annotate
    :type dataset: VideoDataset
    :return: The full dataset
    :rtype: Dataset
    """
    frames = dataset.video_data
    mouse_supplier = EditMouseSupplier(len(frames))
    render_supplier = RenderCrossSupplier(mouse_supplier)

    show_frames(dataset.video_data, 'annotate dataset', mouse_callback=mouse_supplier, frame_callback=render_supplier)

    return Dataset(dataset.video_data, mouse_supplier.annotations)
