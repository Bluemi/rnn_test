from data.data import DatasetPlaceholder, chose_dataset_placeholder, DataInfo
from util.show_frames import show_frames


def show_dataset(args):
    placeholders = DatasetPlaceholder.list_database(args.database_directory)
    joined_dataset_infos = DataInfo.join(map(lambda ds_placeholder: ds_placeholder.data_info, placeholders))
    print('num samples: {}'.format(joined_dataset_infos.num_samples))
    dataset_placeholder = chose_dataset_placeholder(placeholders)

    dataset = dataset_placeholder.load()

    show_frames(dataset)
