import numpy as np

from data.data import Dataset, VideoDataset
from util.images import draw_cross
from util.images.draw_functions import draw_brighter
from util.util import KeyCodes, ACTION_RIGHT_KEYS, ACTION_LEFT_KEYS, RenderWindow


class ShowFramesState:
    def __init__(self, num_frames):
        self.current_index = 0
        self.wait_key_duration = 0
        self.running = True
        self.num_frames = num_frames

    def inc_index(self):
        if self.current_index < self.num_frames - 1:
            self.current_index += 1
        else:
            print('end of video', flush=True)

    def dec_index(self):
        if self.current_index >= 1:
            self.current_index -= 1
        else:
            print('begin of video', flush=True)


def default_key_callback(frames_state, key):
    """
    Changes the frames_state, depending on key.

    :param frames_state: The ShowFramesState object to handle
    :type frames_state: ShowFramesState
    :param key: The pressed key
    :type key: int
    :return: True, if the key was applied otherwise False
    :rtype: bool
    """
    if key == KeyCodes.ESCAPE_KEY:
        frames_state.running = False
    elif key in ACTION_RIGHT_KEYS:
        frames_state.inc_index()
    elif key in ACTION_LEFT_KEYS:
        frames_state.dec_index()
    else:
        return False
    return True


def default_mouse_callback(*_args):
    pass


def default_frame_callback(frame_state, frames):
    """
    Returns the current frame.

    :param frame_state: The current frame state
    :type frame_state: ShowFramesState
    :param frames: The frames to show
    :type frames: list[np.ndarray] or np.ndarray
    :return: The current frame
    :rtype: np.ndarray
    """
    return frames[frame_state.current_index]


class RenderAnnotationsSupplier:
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

        height = current_frame.shape[0]
        width = current_frame.shape[1]

        rel_y, rel_x = self.annotations[index]

        if 0 < rel_y < 1 and 0 < rel_x < 1:
            y = int(rel_y * height)
            x = int(rel_x * width)

            draw_cross(current_frame, (y, x), draw_function=draw_brighter)

        return current_frame


def show_frames(data_source, window_title='frames', key_callback=None, mouse_callback=None, render_callback=None):
    """
    Shows the given frames

    :param data_source: The source of the frames to show. Can be a numpy.ndarray, list[numpy.ndarray], Dataset or
                        VideoDataset
    :type data_source: list[np.ndarray] or np.ndarray or Dataset or VideoDataset
    :param window_title: The title of the window
    :type window_title: str
    :param key_callback: Callback for every keystroke. Should take the current frame_state object, as well as the
                         pressed key
    :type key_callback: Callable or None
    :param mouse_callback: Callback for mouse movements. Should take the current frame_state object, as well as the
    :type mouse_callback: Callable or None
    :param render_callback: Callable that returns the ndarray to render
    :type render_callback: Callable or None

    :raise TypeError: If data_source has invalid type
    """
    if isinstance(data_source, np.ndarray):
        frames = data_source
    elif isinstance(data_source, list):
        for data_source_element in data_source:
            if not isinstance(data_source_element, np.ndarray):
                raise TypeError('Cant show frames for list of type "{}"'.format(type(data_source_element).__name__))
        frames = data_source
    elif isinstance(data_source, Dataset):
        frames = data_source.video_data
        if render_callback is None:
            render_callback = RenderAnnotationsSupplier(data_source.annotation_data)
    elif isinstance(data_source, VideoDataset):
        frames = data_source.video_data
    else:
        raise TypeError('Cant show frames for data source of type "{}"'.format(type(data_source).__name__))

    _show_frames_impl(frames, window_title, key_callback, mouse_callback, render_callback)


def _show_frames_impl(frames, window_title='frames', key_callback=None, mouse_callback=None, frame_callback=None):
    """
    Shows the given frames

    :param frames: The frames to show
    :type frames: list[np.ndarray] or np.ndarray
    :param window_title: The title of the window
    :type window_title: str
    :param key_callback: Callback for every keystroke. Should take the current frame_state object, as well as the
                         pressed key
    :type key_callback: Callable or None
    :param mouse_callback: Callback for mouse movements. Should take the current frame_state object, as well as the
    :type mouse_callback: Callable or None
    :param frame_callback: Callable that returns the ndarray to render
    :type frame_callback: Callable or None

    :return: Returns the control in its end state
    :rtype: ShowFramesState
    """
    frames_state = ShowFramesState(len(frames))

    if key_callback is None:
        key_callback = default_key_callback

    if mouse_callback is None:
        mouse_callback = default_mouse_callback

    if frame_callback is None:
        frame_callback = default_frame_callback

    window = RenderWindow(window_title)
    if mouse_callback is not None:
        window.set_mouse_callback(mouse_callback, frames_state)

    while frames_state.running:
        key = window.show_frame(frame_callback(frames_state, frames), wait_key_duration=frames_state.wait_key_duration)
        key_callback(frames_state, key)

    window.close()

    return frames_state
