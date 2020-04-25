from typing import Iterable
import random

import tensorflow as tf

from data.data import AnnotatedDataset, DatasetPlaceholder, DataInfo
from data.preprocessing import scale_to, no_preprocessing
from model.conv_model import create_compiled_conv_model
from util.show_frames import show_frames, ZoomAnnotationsRenderer


BATCH_SIZE = 32
NUM_EPOCHS = 20

IMAGE_SIZE = (256, 256)
RESOLUTION = (*IMAGE_SIZE, 3)


def _get_tf_dataset(dataset_placeholders, batch_size=BATCH_SIZE, preprocessing=no_preprocessing):
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
            for video_data, annotation_data in zip(annotated_dataset.video_data, annotated_dataset.annotation_data):
                yield preprocessing(video_data, annotation_data)

    return tf.data.Dataset.from_generator(
        _dataset_gen,
        (tf.float32, tf.float32),
        (tf.TensorShape(RESOLUTION), tf.TensorShape([2]))
    ) \
        .take((joined_data_info.num_samples // batch_size) * batch_size)\
        .shuffle(512*2)\
        .batch(batch_size, drop_remainder=True)\
        .repeat()\
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


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
    train_dataset = _get_tf_dataset(train_dataset_placeholders, preprocessing=scale_to(IMAGE_SIZE))

    eval_dataset = AnnotatedDataset.concatenate(map(AnnotatedDataset.from_placeholder, eval_dataset_placeholders))
    eval_dataset = eval_dataset.scale(IMAGE_SIZE)

    print('num train samples: {}'.format(joined_train_data_info.num_samples))
    print('num eval samples: {}'.format(eval_dataset.get_num_samples()))

    print('eval shape: {}'.format(eval_dataset.get_resolution()))

    model = create_compiled_conv_model(RESOLUTION)
    model.summary()
    model.fit(
        train_dataset,
        steps_per_epoch=joined_train_data_info.num_samples // BATCH_SIZE,
        validation_data=(eval_dataset.video_data, eval_dataset.annotation_data),
        validation_steps=eval_dataset.get_num_samples(),
        epochs=NUM_EPOCHS,
        verbose=True,
    )

    if args.show:
        annotations = model.predict(x=eval_dataset.video_data)

        show_frames(
            eval_dataset,
            render_callback=ZoomAnnotationsRenderer(
                [eval_dataset.annotation_data, annotations],
                eval_dataset.get_resolution()
            )
        )


def _join_dataset_placeholder_infos(dataset_placeholders):
    """
    Returns the joined DataInfo of the given placeholders.

    :param dataset_placeholders: Iterable of DatasetPlaceholders to join
    :type dataset_placeholders: Iterable[DatasetPlaceholder]
    :return: The joined DataInfo of the given placeholders
    :rtype: DataInfo
    """
    return DataInfo.join(map(lambda dataset_placeholder: dataset_placeholder.data_info, dataset_placeholders))
