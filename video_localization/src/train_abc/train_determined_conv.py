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
        model = create_uncompiled_conv_model(RESOLUTION)
        model = self.context.wrap_model(model)
        model.compile(
            loss=keras.losses.MeanSquaredError(),
            optimizer=keras.optimizers.Adam(learning_rate=0.0003),
            metrics=[keras.metrics.MeanAbsoluteError()]
        )
        return model

    def build_training_data_loader(self) -> InputData:
        return _get_tf_dataset(self.train_dataset_placeholders, augmentation=create_augmentation())

    def build_validation_data_loader(self) -> InputData:
        return _get_tf_dataset(self.eval_dataset_placeholders)

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
