from data.data import DatasetPlaceholder, Dataset
from model.conv_model import create_compiled_conv_model
from util.show_frames import show_frames


def train_conv_model(args):
    dataset_placeholders = DatasetPlaceholder.list_database(
        args.database_directory, dataset_filter=DatasetPlaceholder.is_full_dataset
    )

    datasets = list(map(Dataset.from_placeholder, dataset_placeholders))

    train_dataset = Dataset.concatenate(datasets)

    model = create_compiled_conv_model(datasets[0].get_resolution())
    model.fit(
        x=train_dataset.video_data,
        y=train_dataset.get_normalized_annotation_data(),
        batch_size=5,
        epochs=3,
        verbose=True
    )
