import numpy as np


def normalize_color(color):
    """
    Normalizes a color to be between 0.0 and 1.0

    :param color: The color to normalize
    :type color: np.ndarray
    :return: The normalized color
    :rtype: np.ndarray
    """
    return np.minimum(np.maximum(color, 0), 1)


def create_draw_brighter(change):
    """
    Creates a draw brighter function, with the given change.

    :param change: The value to add to the color.
    :type change: float or np.ndarray
    :return: A callable that can be used to change the color of an image
    :rtype: Callable[[np.ndarray], np.ndarray]
    """
    def _draw_brighter(color):
        """
        Returns a color that is slightly brighter than the given color.

        :param color: The color to modify
        :type color: np.ndarray
        :return: Brighter version of the given color
        """
        return normalize_color(color + change)

    return _draw_brighter


def create_draw_addition(change):
    """
    Creates a draw addition function, with the given change.

    :param change: The change to add to the color given
    :type change: float or np.ndarray
    :return: A callable that can be used to change the color of an image
    :rtype: Callable[[np.ndarray], np.ndarray]
    """
    def _draw_addition(color):
        """
        Returns a color that has changed in the given direction.

        :param color: The color to change
        :type color: np.ndarray
        :return: The changed color
        :rtype: np.ndarray
        """
        new_color = color + change
        max_c = np.max(new_color)
        if max_c > 1.0:
            new_color -= max_c - 1.0
        return normalize_color(new_color)

    return _draw_addition


draw_brighter = create_draw_brighter(0.2)
draw_addition = create_draw_addition(0.2)


def dark_version(color):
    """
    Returns the negative version of the given color. The result can include negative numbers.

    :param color: The color to invert
    :type color: np.ndarray
    :return: The dark color
    :rtype: np.ndarray
    """
    return color - np.max(color)
