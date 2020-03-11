import numpy as np


def is_in_range(frame, position):
    """
    Checks whether the given position is inside the frame

    :param frame: The frame to check against. Should have dtype float with values between 0 and 1
    :type frame: np.ndarray
    :param position: The position in absolute coordinates as [y, x]
    :type position: tuple[int, int]
    """
    return 0 <= position[0] < frame.shape[0] and 0 <= position[1] < frame.shape[1]


def draw_point(frame, position, draw_function=None):
    """
    Draws a point into the given frame at the given position.

    :param frame: The frame to draw to. Should have dtype float with values between 0 and 1
    :type frame: np.ndarray
    :param position: The position in absolute coordinates as [y, x]
    :type position: tuple[int, int]
    :param draw_function: The function to calculate the next color. Should accept the old pixel color and returns the
                          new pixel color. Defaults to drawing white.
    :type draw_function: Callable[[float], float] or None
    """
    if is_in_range(frame, position):
        if draw_function is None:
            frame[position] = np.array([1, 1, 1])
        else:
            frame[position] = draw_function(frame[position])


def draw_cross(frame, position, size=3, draw_function=None):
    """
    Draws a cross into the given frame at the given position.

    :param frame: The frame to draw to
    :type frame: np.ndarray
    :param position: The position in absolute coordinates
    :type position: tuple[int, int]
    :param size: The size of the cross. The cross will be size*2 + 1 pixels wide and high
    :param draw_function: The function to calculate the next color. Should accept the old pixel color and returns the
                          new pixel color. Defaults to drawing white.
    :type draw_function: Callable[[float], float] or None
    """
    for y in range(size*2+1):
        pos = (position[0]-size+y, position[1])
        draw_point(frame, pos, draw_function)

    for x in range(size*2 + 1):
        pos = (position[0], position[1]-size+x)
        if x != size:
            draw_point(frame, pos, draw_function)
