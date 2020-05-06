import cv2
import tensorflow as tf


def no_preprocessing(image, annotation):
    return image, annotation


def chain(steps):
    """
    Creates a preprocessing function that executes the given steps.

    :param steps: The steps to execute for preprocessing
    :type steps: List[Callable[[numpy.ndarray, numpy.ndarray], tuple[numpy.ndarray, numpy.ndarray]]]
    :return: Returns a preprocessing function that executes the given steps.
    :rtype: Callable[[numpy.ndarray, numpy.ndarray], tuple[numpy.ndarray, numpy.ndarray]]
    """
    def _chain(image, annotation):
        for step in steps:
            image, annotation = step(image, annotation)
        return image, annotation

    return _chain


def scale_to(size):
    """
    Creates a preprocessing function that scaled the images to the given size

    :param size: The size after the preprocessing given as tuple (height, width)
    :type size: tuple[int, int]
    :return: Returns a preprocessing function that scales the image to the given size
    :rtype: Callable[[numpy.ndarray, numpy.ndarray], tuple[numpy.ndarray, numpy.ndarray]]
    """
    def _scale(image_data, annotation_data):
        return cv2.resize(image_data, (size[1], size[0])), annotation_data

    return _scale


def random_brightness(brightness_variance):
    """
    Creates a preprocessing function that adds random brightness to the image.

    :param brightness_variance: The brightness variance of the resulting image
    :type brightness_variance: float
    :return: A preprocessing function that adds random brightness to the image
    :rtype: Callable[[numpy.ndarray, numpy.ndarray], tuple[numpy.ndarray, numpy.ndarray]]
    """
    def _random_brightness(image, annotation):
        return tf.image.random_brightness(image, brightness_variance), annotation

    return _random_brightness
