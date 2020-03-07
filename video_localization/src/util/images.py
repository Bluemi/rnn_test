import numpy as np


def draw_point(frame, position):
    """
    Draws a point into the given frame at the given position.

    :param frame: The frame to draw to
    :type frame: np.ndarray
    :param position: The position in absolute coordinates
    :type position: tuple[int]
    :return:
    """
    frame[position] = np.array([256, 256, 256])
