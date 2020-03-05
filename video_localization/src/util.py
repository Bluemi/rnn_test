import time
from collections import deque, Callable

import cv2
import numpy as np


ESCAPE_KEY = 27
ENTER_KEY = 13
LEFT_KEY = 81
RIGHT_KEY = 83
A_KEY = 97
E_KEY = 101


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

    def set_mouse_callback(self, mouse_callback: Callable):
        """
        Sets the mouse callback for this window.

        :param mouse_callback: A function that will be called if the mouse moves
        :type mouse_callback: Callable
        """
        cv2.setMouseCallback(self.title, mouse_callback)

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
        if key == ESCAPE_KEY:
            self.running = False
        elif key == RIGHT_KEY:
            self.inc_index()
        elif key == LEFT_KEY:
            self.dec_index()
        else:
            return False
        return True

    def mouse_callback(self, event_type, x, y, *args):
        pass


class EditFramesControl(ShowFramesControl):
    def __init__(self, max_index):
        super(EditFramesControl, self).__init__(max_index)
        self.start_index = 0
        self.end_index = max_index

    def apply_key(self, key):
        if super().apply_key(key):
            return True

        if key == ENTER_KEY:
            self.running = False
        if key == A_KEY:
            self.start_index = self.current_index
            print('set start index = {}'.format(self.start_index))
        elif key == E_KEY:
            self.end_index = self.current_index + 1
            print('set end index = {}'.format(self.end_index))
        else:
            return False

        return True


class PointHandsControl(ShowFramesControl):
    def __init__(self, max_index):
        super(PointHandsControl, self).__init__(max_index)
        self.points = np.empty((max_index, 2), dtype=np.float)
        self.x = None
        self.y = None

    def _has_x_y(self):
        return self.x is not None and self.y is not None

    def _set_point(self):
        if self._has_x_y():
            self.points[self.current_index][0] = self.x
            self.points[self.current_index][1] = self.y
            self.inc_index()
        else:
            print('Could not set point, because x, y was not defined.', flush=True)

    def mouse_callback(self, event_type, x, y, *args):
        if event_type == 0:
            self.x = x
            self.y = y
        elif event_type == 1:
            self._set_point()


def show_frames(frames, control=None, window_title='frames'):
    """

    :param frames: The frames to show
    :type frames: list[np.ndarray] or np.ndarray
    :param control: A ShowFramesControl to manage the frames
    :type control: ShowFramesControl
    :param window_title: The title of the window
    :type window_title: str

    :return: Returns the control in its end state
    :rtype: ShowFramesControl
    """
    window = RenderWindow(window_title)

    if control is None:
        control = ShowFramesControl(len(frames))

    window.set_mouse_callback(control.mouse_callback)

    while control.running:
        key = window.show_frame(frames[control.current_index], wait_key_duration=control.wait_key_duration)
        control.apply_key(key)

    window.close()

    return control
