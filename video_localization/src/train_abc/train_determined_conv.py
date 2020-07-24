import tensorflow.keras as keras
from determined.keras import TFKerasTrial, TFKerasTrialContext, InputData

from data.data import DatasetPlaceholder
from model.conv_model import create_uncompiled_conv_model
from train_abc.train_conv_model import RESOLUTION, _join_dataset_placeholder_infos, _get_tf_dataset, create_augmentation

DATA_DIRECTORY = '/data/train_data'


class ConvTrial(TFKerasTrial):
    def __init__(self, context: TFKerasTrialContext):
        super().__init__(context)

        # load dataset placeholders
        self.train_dataset_placeholders, self.eval_dataset_placeholders = ConvTrial.list_placeholders()
        self.joined_train_data_info = _join_dataset_placeholder_infos(self.train_dataset_placeholders)
        self.joined_eval_data_info = _join_dataset_placeholder_infos(self.eval_dataset_placeholders)

    def build_model(self) -> keras.models.Model:
        model = create_uncompiled_conv_model(
            RESOLUTION,
            num_filters1=self.context.get_hparam("num_filters1"),
            num_filters2=self.context.get_hparam("num_filters2"),
            num_filters3=self.context.get_hparam("num_filters3"),
            regularization1=self.context.get_hparam("regularization1"),
            regularization2=self.context.get_hparam("regularization2"),
            regularization3=self.context.get_hparam("regularization3"),
            num_nodes_dense1=self.context.get_hparam("num_nodes_dense1"),
            regularization_dense1=self.context.get_hparam("regularization_dense1"),
            regularization_dense_bias1=self.context.get_hparam("regularization_dense_bias1"),
            regularization_output_bias=self.context.get_hparam("regularization_output_bias"),
        )
        model = self.context.wrap_model(model)
        model.compile(
            loss=keras.losses.MeanSquaredError(),
            optimizer=keras.optimizers.Adam(learning_rate=0.0003),
            metrics=[keras.metrics.MeanAbsoluteError()]
        )
        return model

    def build_training_data_loader(self) -> InputData:
        dataset = _get_tf_dataset(
            self.train_dataset_placeholders,
            augmentation=create_augmentation(),
            cache=False,
            batch=True,
            repeat=False
        )
        dataset = self.context.wrap_dataset(dataset)
        return dataset

    def build_validation_data_loader(self) -> InputData:
        dataset = _get_tf_dataset(
            self.eval_dataset_placeholders,
            cache=False,
            batch=True,
            repeat=False
        )
        dataset = self.context.wrap_dataset(dataset)
        return dataset

    @staticmethod
    def list_placeholders():
        dataset_placeholders = DatasetPlaceholder.list_database(DATA_DIRECTORY, DatasetPlaceholder.is_full_dataset)

        train_dataset_placeholders = []
        eval_dataset_placeholders = []

        for dataset_placeholder in dataset_placeholders:
            if 'eval' in dataset_placeholder.data_info.tags:
                eval_dataset_placeholders.append(dataset_placeholder)
            else:
                train_dataset_placeholders.append(dataset_placeholder)
        return train_dataset_placeholders, eval_dataset_placeholders
