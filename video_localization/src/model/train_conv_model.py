from typing import Iterable
import random

import numpy as np
import tensorflow as tf

from data.data import AnnotatedDataset, DatasetPlaceholder, DataInfo
from data.preprocessing import scale_to, random_brightness, chain
from model.conv_model import create_compiled_conv_model
from util.images import draw_cross
from util.images.draw_functions import create_draw_addition, dark_version
from util.util import RenderWindow, KeyCodes

BATCH_SIZE = 32
NUM_EPOCHS = 20

IMAGE_SIZE = (128, 128)
RESOLUTION = (*IMAGE_SIZE, 3)


def _get_tf_dataset(dataset_placeholders, batch_size=BATCH_SIZE, preprocessing=None):
    """
    Returns a tf Dataset that can be used for training.

    :param dataset_placeholders: The placeholders to use for this dataset
    :type dataset_placeholders: List[DatasetPlaceholder]
    :param preprocessing: A callable that gets called for every image and annotation data
    :type preprocessing: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
    :return: A tensorflow Dataset
    :rtype: tf.data.Dataset
    """
    joined_data_info = _join_dataset_placeholder_infos(dataset_placeholders)

    def _dataset_gen():
        for dataset_placeholder in random.sample(dataset_placeholders, len(dataset_placeholders)):
            annotated_dataset = AnnotatedDataset.from_placeholder(dataset_placeholder, divisible_by=batch_size)
            for image, annotation in zip(annotated_dataset.video_data, annotated_dataset.annotation_data):
                yield preprocessing(image, annotation)

    dataset = tf.data.Dataset.from_generator(
        _dataset_gen,
        (tf.float32, tf.float32),
        (tf.TensorShape(RESOLUTION), tf.TensorShape([2]))
    )
    dataset = dataset.take((joined_data_info.num_samples // batch_size) * batch_size)
    dataset = dataset.cache()
    dataset = dataset.shuffle(joined_data_info.num_samples)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.repeat()
    if preprocessing is not None:
        dataset = dataset.map(preprocessing, 4)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def create_preprocessing():
    scale = scale_to(IMAGE_SIZE)
    brightness = random_brightness(0.2)
    return chain([scale, brightness])


def add_annotation(image, annotation, color):
    y, x = int(annotation[0] * image.shape[0]), int(annotation[1] * image.shape[1])
    if 0 <= annotation[0] <= 1 and 0 <= annotation[1] <= 1 and np.mean(image[y, x]) > 0.4:
        color = dark_version(color)
    draw_cross(image, (y, x), draw_function=create_draw_addition(color))


def show_dataset(dataset, extra_annotations=None):
    render_window = RenderWindow('dataset', (50, 50))
    index = 0
    for sample in dataset:
        image_data, annotation_data = sample
        for image, annotation in zip(image_data, annotation_data):
            image = image.numpy()
            annotation = annotation.numpy()
            add_annotation(image, annotation, np.array([0.0, 0.5, 0.0]))
            if extra_annotations is not None:
                extra_annotation = extra_annotations[index]
                add_annotation(image, extra_annotation, np.array([0.5, 0.0, 0.0]))
            if render_window.show_frame(image) == KeyCodes.ESCAPE:
                return
            index += 1


def train_conv_model(args):
    dataset_placeholders = DatasetPlaceholder.list_database(args.train_data, DatasetPlaceholder.is_full_dataset)

    train_dataset_placeholders = []
    eval_dataset_placeholders = []

    for dataset_placeholder in dataset_placeholders:
        if 'eval' in dataset_placeholder.data_info.tags:
            eval_dataset_placeholders.append(dataset_placeholder)
        else:
            train_dataset_placeholders.append(dataset_placeholder)

    joined_train_data_info = _join_dataset_placeholder_infos(train_dataset_placeholders)
    train_dataset = _get_tf_dataset(train_dataset_placeholders, preprocessing=create_preprocessing())

    # show_dataset(train_dataset)

    joined_eval_data_info = _join_dataset_placeholder_infos(eval_dataset_placeholders)
    eval_dataset = _get_tf_dataset(eval_dataset_placeholders, preprocessing=create_preprocessing())

    print('num train samples: {}'.format(joined_train_data_info.num_samples))
    print('num eval samples: {}'.format(joined_eval_data_info.num_samples))

    model = create_compiled_conv_model(RESOLUTION)
    model.summary()
    model.fit(
        train_dataset,
        steps_per_epoch=joined_train_data_info.num_samples // BATCH_SIZE,
        validation_data=eval_dataset,
        validation_steps=joined_eval_data_info.num_samples // BATCH_SIZE,
        epochs=NUM_EPOCHS,
        verbose=True,
    )

    if args.show:
        annotations = model.predict(x=eval_dataset, steps=joined_eval_data_info.num_samples // BATCH_SIZE)

        show_dataset(eval_dataset, annotations)


def _join_dataset_placeholder_infos(dataset_placeholders):
    """
    Returns the joined DataInfo of the given placeholders.

    :param dataset_placeholders: Iterable of DatasetPlaceholders to join
    :type dataset_placeholders: Iterable[DatasetPlaceholder]
    :return: The joined DataInfo of the given placeholders
    :rtype: DataInfo
    """
    return DataInfo.join(map(lambda dataset_placeholder: dataset_placeholder.data_info, dataset_placeholders))
