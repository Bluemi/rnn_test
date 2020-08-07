import random

import cv2
import numpy as np
import tensorflow as tf

from data.data import AnnotatedDataset, DataInfo
from data.preprocessing import scale_to, random_brightness, RandomTransformer, chain
from util.images import draw_cross
from util.images.draw_functions import dark_version, create_draw_addition
from util.util import RenderWindow, KeyCodes


BATCH_SIZE = 32
IMAGE_SIZE = (128, 128)
RESOLUTION = (*IMAGE_SIZE, 3)
TRANSFORM_SCALE = 1.05
TRANSFORM_TRANSLATION = 7


def get_tf_dataset(
        dataset_placeholders,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        augmentation=None,
        cache=True,
        batch=True,
        repeat=True,
        shuffle=True,
):
    """
    Returns a tf Dataset that can be used for training.

    :param dataset_placeholders: The placeholders to use for this dataset
    :type dataset_placeholders: List[DatasetPlaceholder]
    :param image_size: The scale for the images given as tuple (height, width)
    :type image_size: tuple[int, int]
    :param augmentation: A callable that gets called for every image and annotation data
    :type augmentation: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
    :param cache: Whether to cache the dataset or not
    :type cache: bool
    :param batch: Whether to batch the dataset or not
    :type batch: bool
    :param repeat: Whether to repeat the dataset or not
    :type repeat: bool
    :param shuffle: Whether to shuffle the dataset or not
    :type shuffle: bool
    :return: A tensorflow Dataset
    :rtype: tf.data.Dataset
    """
    joined_data_info = _join_dataset_placeholder_infos(dataset_placeholders)
    scale_func = scale_to(image_size)

    def _dataset_gen():
        for dataset_placeholder in random.sample(dataset_placeholders, len(dataset_placeholders)):
            annotated_dataset = AnnotatedDataset.from_placeholder(dataset_placeholder, divisible_by=batch_size)
            for image, annotation in zip(annotated_dataset.video_data, annotated_dataset.annotation_data):
                yield scale_func(image, annotation)

    dataset = tf.data.Dataset.from_generator(
        _dataset_gen,
        (tf.float32, tf.float32),
        (tf.TensorShape(RESOLUTION), tf.TensorShape([2]))
    )
    dataset = dataset.take((joined_data_info.num_samples // batch_size) * batch_size)
    if cache:
        dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(joined_data_info.num_samples)
    if batch:
        dataset = dataset.batch(batch_size, drop_remainder=True)
    if repeat:
        dataset = dataset.repeat()
    if augmentation is not None:
        dataset = dataset.map(augmentation, 4)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def create_augmentation():
    brightness = random_brightness(0.2)
    transformer = RandomTransformer(TRANSFORM_SCALE, TRANSFORM_TRANSLATION)
    return chain([brightness, transformer])


def add_annotation(image, annotation, color):
    y, x = int(annotation[0] * image.shape[0]), int(annotation[1] * image.shape[1])
    if 0 <= annotation[0] <= 1 and 0 <= annotation[1] <= 1 and np.mean(image[y, x]) > 0.4:
        color = dark_version(color)
    draw_cross(image, (y, x), draw_function=create_draw_addition(color))


def show_dataset(dataset, extra_annotations=None, size=(512, 512)):
    render_window = RenderWindow('dataset', (50, 50))
    index = 0
    for image_data, annotation_data in dataset:
        for image, annotation in zip(image_data, annotation_data):
            if not isinstance(image, np.ndarray):
                image = image.numpy()
            if size is not None:
                image = cv2.resize(image, size)
            if not isinstance(annotation, np.ndarray):
                annotation = annotation.numpy()
            add_annotation(image, annotation, np.array([0.0, 0.5, 0.0]))
            if extra_annotations is not None:
                extra_annotation = extra_annotations[index]
                add_annotation(image, extra_annotation, np.array([0.0, 0.0, 1.0]))
            if render_window.show_frame(image) == KeyCodes.ESCAPE:
                return
            index += 1


def _join_dataset_placeholder_infos(dataset_placeholders):
    """
    Returns the joined DataInfo of the given placeholders.

    :param dataset_placeholders: Iterable of DatasetPlaceholders to join
    :type dataset_placeholders: Iterable[DatasetPlaceholder]
    :return: The joined DataInfo of the given placeholders
    :rtype: DataInfo
    """
    return DataInfo.join(map(lambda dataset_placeholder: dataset_placeholder.data_info, dataset_placeholders))