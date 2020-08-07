import os
import time

import numpy as np
import tensorflow as tf

from data.data import DatasetPlaceholder
from model.conv_model import create_compiled_conv_model
from train.train_util import get_tf_dataset, create_augmentation, show_dataset, _join_dataset_placeholder_infos, \
    RESOLUTION, BATCH_SIZE
from train.hyperparameter_sets import hyperparameter_set_1 as hyperparameter_set

NUM_EPOCHS = 2

NUM_SHOW_BATCHES = 2

CHECKPOINT_DIR = 'models'
MODEL_TIME_FORMAT = '%H_%M_%S__%d_%m_%Y'


def get_next_model_dir():
    model_dir = 'model_{}'.format(time.strftime(MODEL_TIME_FORMAT))
    return os.path.join(CHECKPOINT_DIR, model_dir)


def train_conv_model(args):
    dataset_placeholders = DatasetPlaceholder.list_database(args.train_data, DatasetPlaceholder.is_full_dataset)

    train_dataset_placeholders = []
    val_dataset_placeholders = []

    for dataset_placeholder in dataset_placeholders:
        if 'val' in dataset_placeholder.data_info.tags:
            val_dataset_placeholders.append(dataset_placeholder)
        else:
            train_dataset_placeholders.append(dataset_placeholder)

    joined_train_data_info = _join_dataset_placeholder_infos(train_dataset_placeholders)
    train_dataset = get_tf_dataset(train_dataset_placeholders, augmentation=create_augmentation())

    joined_val_data_info = _join_dataset_placeholder_infos(val_dataset_placeholders)
    val_dataset = get_tf_dataset(val_dataset_placeholders, augmentation=create_augmentation())

    print('num train samples: {}'.format(joined_train_data_info.num_samples))
    print('num val samples: {}'.format(joined_val_data_info.num_samples))

    model = create_compiled_conv_model(RESOLUTION, **hyperparameter_set)
    model.summary()

    # checkpointing
    model_dir = get_next_model_dir()
    os.makedirs(model_dir, exist_ok=False)
    model_path = os.path.join(model_dir, 'model_{epoch:03d}-{val_mean_absolute_error:.3f}.hdf5')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        save_weights_only=False,
        monitor='val_mean_absolute_error',
        mode='min',
        save_best_only=True,
        verbose=1
    )
    model.fit(
        train_dataset,
        steps_per_epoch=joined_train_data_info.num_samples // BATCH_SIZE,
        validation_data=val_dataset,
        validation_steps=joined_val_data_info.num_samples // BATCH_SIZE,
        epochs=NUM_EPOCHS,
        verbose=True,
        callbacks=[model_checkpoint_callback],
    )

    if args.show:
        x_data = []

        show_data = []

        for x_batch, y_batch in val_dataset:
            x_data.append(x_batch.numpy())
            show_data.append((x_batch.numpy(), y_batch.numpy()))
            if len(x_data) >= NUM_SHOW_BATCHES:
                break

        x_data = np.array(x_data)
        x_data.resize((BATCH_SIZE*NUM_SHOW_BATCHES, *RESOLUTION))

        annotations = model.predict(x=x_data, steps=NUM_SHOW_BATCHES)

        show_dataset(show_data, annotations)
