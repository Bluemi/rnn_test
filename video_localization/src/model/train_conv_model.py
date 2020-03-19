import tensorflow as tf

from data.data import AnnotatedDataset, DatasetPlaceholder
from model.conv_model import create_compiled_conv_model
from util.show_frames import show_frames, ZoomAnnotationsRenderer


def _get_tf_dataset(database_dir):
    dataset_placeholders = DatasetPlaceholder.list_database(database_dir, DatasetPlaceholder.is_full_dataset)

    def _dataset_gen():
        for dataset_placeholder in dataset_placeholders:
            d_set = AnnotatedDataset.from_placeholder(dataset_placeholder)
            for video_data, annotation_data in zip(d_set.video_data, d_set.annotation_data):
                yield video_data, annotation_data

    dataset = tf.data.Dataset.from_generator(
        _dataset_gen,
        (tf.float32, tf.float32),
        (tf.TensorShape([480, 640, 3]), tf.TensorShape([2]))
    )

    dataset = dataset.batch(30, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def train_conv_model(args):
    train_dataset = _get_tf_dataset(args.train_data)

    resolution = (480, 640, 3)

    eval_dataset = None
    if args.eval_data is not None:
        eval_dataset = _get_tf_dataset(args.eval_data)

    model = create_compiled_conv_model(resolution)

    model.summary()
    model.fit(
        train_dataset,
        validation_data=eval_dataset,
        epochs=10,
        verbose=True
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
