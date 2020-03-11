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


def always_true(_unused):
    return True
