import time
from collections import deque, Callable
from enum import IntEnum

import cv2
import numpy as np


class KeyCodes(IntEnum):
    BACK_SPACE = 8
    ENTER = 13
    ESCAPE = 27
    SPACE = 32
    PLUS = 43
    MINUS = 45
    LEFT = 81
    RIGHT = 83
    END = 87
    A = 97
    B = 98
    C = 99
    E = 101
    H = 104
    I = 105
    J = 106
    K = 107
    L = 108
    N = 110
    P = 112
    T = 116
    U = 117
    W = 119


ACTION_PREVIOUS_KEYS = (KeyCodes.LEFT, KeyCodes.BACK_SPACE, KeyCodes.B)
ACTION_NEXT_KEYS = (KeyCodes.RIGHT, KeyCodes.SPACE, KeyCodes.N)


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
    def __init__(self, title, position=None):
        """
        Creates a new window with the given title

        :param title: The title of the window
        :type title: str
        :param position: The initial position of the window given as (y, x)
        :type position: tuple[int, int] or None
        """
        self.title = title
        cv2.namedWindow(title)
        if position is not None:
            cv2.moveWindow(title, position[1], position[0])

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
