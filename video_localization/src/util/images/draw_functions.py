import numpy as np


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
        return np.minimum(np.maximum(color + change, 0), 1)

    return _draw_brighter


draw_brighter = create_draw_brighter(0.2)
