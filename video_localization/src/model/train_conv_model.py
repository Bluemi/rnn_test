from typing import Iterable

import tensorflow as tf

from data.data import AnnotatedDataset, DatasetPlaceholder, DataInfo
from model.conv_model import create_compiled_conv_model
from util.show_frames import show_frames, ZoomAnnotationsRenderer


BATCH_SIZE = 32
NUM_EPOCHS = 100


def _get_tf_dataset(dataset_placeholders, batch_size=BATCH_SIZE):
    """
    Returns a tf Dataset that can be used for training.

    :param dataset_placeholders: The placeholders to use for this dataset
    :type dataset_placeholders: Iterable[DatasetPlaceholder]
    :return: A tensorflow Dataset
    :rtype: tf.data.Dataset
    """
    joined_data_info = _join_dataset_placeholder_infos(dataset_placeholders)

    def _dataset_gen():
        def _sample_iter():
            for dataset_placeholder in dataset_placeholders:
                annotated_dataset = AnnotatedDataset.from_placeholder(dataset_placeholder, divisible_by=batch_size)
                for video_data, annotation_data in zip(annotated_dataset.video_data, annotated_dataset.annotation_data):
                    yield video_data, annotation_data

        return _sample_iter()

    return tf.data.Dataset.from_generator(
        _dataset_gen,
        (tf.float32, tf.float32),
        (tf.TensorShape(joined_data_info.resolution), tf.TensorShape([2]))
    ) \
        .take((joined_data_info.num_samples // batch_size) * batch_size)\
        .batch(batch_size, drop_remainder=True)\
        .repeat()\
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


def train_conv_model(args):
    dataset_placeholders = DatasetPlaceholder.list_database(args.train_data, DatasetPlaceholder.is_full_dataset)

    train_dataset_placeholders = []
    eval_dataset_placeholders = []

    for dataset_placeholder in dataset_placeholders:
        if 'rudi' in dataset_placeholder.data_info.subjects:
            eval_dataset_placeholders.append(dataset_placeholder)
        else:
            train_dataset_placeholders.append(dataset_placeholder)

    joined_train_data_info = _join_dataset_placeholder_infos(train_dataset_placeholders)
    train_dataset = _get_tf_dataset(train_dataset_placeholders)

    eval_dataset = AnnotatedDataset.concatenate(map(AnnotatedDataset.from_placeholder, eval_dataset_placeholders))
    joined_eval_data_info = _join_dataset_placeholder_infos(eval_dataset_placeholders)

    model = create_compiled_conv_model(joined_train_data_info.resolution)
    model.summary()
    model.fit(
        train_dataset,
        steps_per_epoch=joined_train_data_info.num_samples // BATCH_SIZE,
        validation_data=(eval_dataset.video_data, eval_dataset.annotation_data),
        validation_steps=joined_eval_data_info.num_samples if joined_eval_data_info else None,
        epochs=NUM_EPOCHS,
        verbose=True,
    )

    if args.show:
        show_data_dir = args.eval_data or args.train_data

        show_dataset = AnnotatedDataset.load_database(show_data_dir)

        annotations = model.predict(x=show_dataset.video_data)

        show_frames(
            show_dataset,
            render_callback=ZoomAnnotationsRenderer(
                [show_dataset.annotation_data, annotations],
                show_dataset.get_resolution()
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
