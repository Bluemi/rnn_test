import time
from collections import deque, Callable
from enum import IntEnum

import cv2
import numpy as np


class KeyCodes(IntEnum):
    ESCAPE_KEY = 27
    ENTER_KEY = 13
    SPACE_KEY = 32
    LEFT_KEY = 81
    RIGHT_KEY = 83
    L_KEY = 108
    H_KEY = 104
    A_KEY = 97
    E_KEY = 101


ACTION_LEFT_KEYS = (KeyCodes.LEFT_KEY, KeyCodes.H_KEY)
ACTION_RIGHT_KEYS = (KeyCodes.RIGHT_KEY, KeyCodes.L_KEY, KeyCodes.SPACE_KEY)


class FPS:
    def __init__(self, observation_period=1):
        """
        Creates a new FPS counter.

        :param observation_period: The duration that is considered to calculate the fps in seconds
        :type observation_period: float
        """
        self.times = deque()
        self.observation_period = observation_period

    def update(self):
        """
        Should be called once per frame.
        """
        self.times.append(time.clock())

    def get_fps(self):
        """
        Returns the fps of calling the update method in the last observation period.

        :return: the fps of calling the update method in the last observation period.
        :rtype: float
        """
        time_limit = time.clock() - self.observation_period
        while self.times and self.times[0] < time_limit:
            self.times.popleft()

        if len(self.times) <= 1:
            return 0

        return 1 / np.mean(np.diff(self.times))


class RenderWindow:
    def __init__(self, title):
        """
        Creates a new window with the given title

        :param title: The title of the window
        :type title: str
        """
        self.title = title
        cv2.namedWindow(title)

    def show_frame(self, frame, wait_key_duration=0):
        """
        Shows the given frame.

        :param frame: The frame to show
        :type frame: np.ndarray
        :param wait_key_duration: The duration to wait for the next keystroke in milliseconds. Defaults to 0, which
                                  means it waits until a key is pressed.
        :type wait_key_duration: int

        :return: The key that was pressed during showing the window
        :rtype: int
        """
        cv2.imshow(self.title, frame)
        return cv2.waitKey(wait_key_duration)

    def set_mouse_callback(self, mouse_callback: Callable, param=None):
        """
        Sets the mouse callback for this window.

        :param mouse_callback: A function that will be called if the mouse moves
        :type mouse_callback: Callable
        :param param: Additional parameter for the mouse callback
        """
        cv2.setMouseCallback(self.title, mouse_callback, param)

    def close(self):
        """
        Closes the window.
        """
        cv2.destroyWindow(self.title)


class ShowFramesControl:
    def __init__(self, max_index):
        self.current_index = 0
        self.max_index = max_index
        self.wait_key_duration = 0
        self.running = True

    def inc_index(self):
        if self.current_index < self.max_index - 1:
            self.current_index += 1
        else:
            print('end of video', flush=True)

    def dec_index(self):
        if self.current_index >= 1:
            self.current_index -= 1
        else:
            print('begin of video', flush=True)

    def apply_key(self, key):
        if key == KeyCodes.ESCAPE_KEY:
            self.running = False
        elif key in ACTION_RIGHT_KEYS:
            self.inc_index()
        elif key in ACTION_LEFT_KEYS:
            self.dec_index()
        else:
            return False
        return True

    def mouse_callback(self, event_type, x, y, *args):
        pass


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


def show_frames(frames, window_title='frames', key_callback=None, mouse_callback=None, frame_callback=None):
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
    :rtype: ShowFramesControl
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
