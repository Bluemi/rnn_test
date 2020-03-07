from data.data import DatasetPlaceholder, Dataset


def train_conv_model(args):
    dataset_placeholders = DatasetPlaceholder.list_database(
        args.database_directory, dataset_filter=DatasetPlaceholder.is_full_dataset
    )

    datasets = list(map(
        lambda dataset_placeholder: Dataset.from_placeholder(dataset_placeholder), dataset_placeholders
    ))

    print(datasets)
