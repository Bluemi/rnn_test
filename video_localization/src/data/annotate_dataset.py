import numpy as np

from data.data import DatasetPlaceholder, chose_dataset_placeholder, VideoDataset
from util.images import draw_cross
from util.images.draw_functions import draw_brighter
from util.show_frames import show_frames, RenderAnnotationsSupplier


def annotate_dataset(args):
    dataset_placeholders = DatasetPlaceholder.list_database(args.database_directory)
    video_dataset_placeholders = list(filter(DatasetPlaceholder.is_video_dataset, dataset_placeholders))

    dataset_placeholder = chose_dataset_placeholder(video_dataset_placeholders)

    source_dataset = dataset_placeholder.load()

    annotations = annotate_frames(source_dataset)

    np.save(dataset_placeholder.get_annotations_path(), annotations)


class EditMouseSupplier:
    def __init__(self, num_samples, resolution):
        """
        Creates a new EditMouseSupplier.

        :param num_samples: The number of samples
        :type num_samples: int
        :param resolution: The resolution of the video, given as [height, width, depth]. depth is optional
        :type resolution: tuple[int, int, int] or tuple[int, int]
        """
        print('resolution: {}'.format(resolution))
        self.annotations = np.zeros((num_samples, 2), dtype=np.float) + np.nan
        self.resolution = resolution[:2]

    def _set_point(self, frames_state, position):
        self.annotations[frames_state.current_index][0] = position[0] / self.resolution[0]
        self.annotations[frames_state.current_index][1] = position[1] / self.resolution[1]

    def is_point_set(self, index):
        return not (np.isnan(self.annotations[index][0]) or np.isnan(self.annotations[index][1]))

    def __call__(self, event_type, x, y, _unused1, frames_state):
        """
        Sets the annotations, if mouse is pressed.

        :param event_type: The mouse event type. 0 -> Mouse Move, 1 -> Left Mouse Click
        :param x: The x position of the mouse
        :param y: The y position of the mouse
        :param frames_state: The current frames_state object
        """
        if event_type == 1:
            self._set_point(frames_state, (y, x))


def annotate_frames(dataset):
    """
    Returns the full dataset.

    :param dataset: The dataset to annotate
    :type dataset: VideoDataset
    :return: The annotations made
    :rtype: np.ndarray
    """
    frames = dataset.video_data
    mouse_supplier = EditMouseSupplier(len(frames), dataset.get_resolution())
    render_supplier = RenderAnnotationsSupplier(mouse_supplier.annotations)

    show_frames(dataset.video_data, 'annotate dataset', mouse_callback=mouse_supplier, render_callback=render_supplier)

    return mouse_supplier.annotations
