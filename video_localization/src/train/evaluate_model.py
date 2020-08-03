import numpy as np
import tensorflow as tf

from data.data import DatasetPlaceholder
from train.train_conv_model import get_tf_dataset, show_dataset

NUM_SHOW_BATCHES = 10
BATCH_SIZE = 32
RESOLUTION = (128, 128, 3)


def evaluate_model(args):
    dataset_placeholders = DatasetPlaceholder.list_database(args.train_data, DatasetPlaceholder.is_full_dataset)
    dataset_placeholders = list(filter(lambda dp: 'eval' in dp.data_info.tags, dataset_placeholders))

    model = tf.keras.models.load_model(args.model)

    if args.show:
        dataset = get_tf_dataset(dataset_placeholders, repeat=False, batch=True, shuffle=False)

        show_data = []

        x_data = []
        for x_batch, y_batch in dataset:
            show_data.append((x_batch.numpy(), y_batch.numpy()))
            x_data.append(x_batch.numpy())
            if len(x_data) >= NUM_SHOW_BATCHES:
                break

        x_data = np.array(x_data)
        x_data.resize((BATCH_SIZE * NUM_SHOW_BATCHES, *RESOLUTION))

        annotations = model.predict(x=x_data, steps=NUM_SHOW_BATCHES)

        show_dataset(show_data, annotations)
    else:
        dataset = get_tf_dataset(dataset_placeholders, repeat=False)

        result = model.evaluate(dataset)

        print(dict(zip(model.metrics_names, result)))

