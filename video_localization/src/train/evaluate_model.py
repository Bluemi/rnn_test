import tensorflow as tf

from data.data import DatasetPlaceholder
from train.train_conv_model import get_tf_dataset


def evaluate_model(args):
    dataset_placeholders = DatasetPlaceholder.list_database(args.train_data, DatasetPlaceholder.is_full_dataset)
    dataset_placeholders = list(filter(lambda dp: 'eval' in dp.data_info.tags, dataset_placeholders))

    dataset = get_tf_dataset(dataset_placeholders, repeat=False)

    model = tf.keras.models.load_model(args.model)

    result = model.evaluate(dataset)

    print(dict(zip(model.metrics_names, result)))
