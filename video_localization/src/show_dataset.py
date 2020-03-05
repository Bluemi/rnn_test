from data import chose_dataset
from util import show_frames


def show_dataset(args):
    dataset = chose_dataset(args.database_directory)

    show_frames(dataset)


