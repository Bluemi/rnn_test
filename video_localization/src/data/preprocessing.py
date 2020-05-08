import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


def _sign_without_zero(v):
    return tf.sign(tf.sign(v) + 0.5)


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
        return tf.image.resize(image_data, size), annotation_data

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


class RandomTransformer:
    def __init__(self, scale, translation):
        self._min, self._range = self._calculate_matrices(scale, translation)

    @staticmethod
    def _calculate_matrices(scale, translation):
        """
        Calculates the _min and _range member
        """
        min_mat = np.float32([
            1.0 / scale,
            0.0,
            -translation,
            0.0,
            1.0 / scale,
            -translation,
            0.0,
            0.0
        ])
        max_mat = np.float32([
            scale,
            0.0,
            translation,
            0.0,
            scale,
            translation,
            0.0,
            0.0
        ])
        range_mat = max_mat - min_mat
        return min_mat, range_mat

    @staticmethod
    def _get_transformed_annotations(x_, y_, a0, a2, b1, b2):
        """
        Transforms the point (x_, y_) from the input image transformed by the transformation matrix given by the other
        parameters.
        Tensors can also be used for the arguments.
        :return: The transformed point
        """
        x = (x_ - a2) / a0
        y = (y_ - b2) / b1
        return x, y

    @staticmethod
    def _transform_annotations(annotations, transformation_matrix, image_size):
        """
        Transforms the given annotations with the given transformation matrix.

        :param annotations: The annotations to transform
        :param transformation_matrix: The transformation matrix
        """
        new_annotations = RandomTransformer._get_transformed_annotations(
            annotations[:, 1] * image_size[1],
            annotations[:, 0] * image_size[0],
            transformation_matrix[:, 0],
            # we skip 1, 3, 6 and 7, because they are always zero
            transformation_matrix[:, 2],
            # we skip 1, 3, 6 and 7, because they are always zero
            transformation_matrix[:, 4],
            transformation_matrix[:, 5],
            # we skip 1, 3, 6 and 7, because they are always zero
            # we skip 1, 3, 6 and 7, because they are always zero
        )
        new_annotations = (new_annotations[1] / image_size[1], new_annotations[0] / image_size[0])
        return tf.stack(new_annotations, axis=-1)

    def __call__(self, images, annotations):
        """
        Applies a random transformation to the image and the annotation

        :param images: The images to transform
        :param annotations: The annotations to transform
        :return:
        """
        batch_size = images.shape[0]
        transformation_matrix = self._min + self._range * tf.random.uniform((batch_size, 8))
        transformed_images = tfa.image.transform(images, transformation_matrix)
        transformed_annotations = RandomTransformer._transform_annotations(
            annotations,
            transformation_matrix,
            (images.shape[1], images.shape[2])
        )
        return transformed_images, transformed_annotations
