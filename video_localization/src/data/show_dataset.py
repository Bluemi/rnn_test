from data.data import chose_dataset_placeholder_from_database
from util.show_frames import show_frames, ZoomRenderer


def show_dataset(args):
    dataset_placeholder = chose_dataset_placeholder_from_database(args.database_directory)

    dataset = dataset_placeholder.load()

    show_frames(dataset, render_callback=ZoomRenderer(enable_cross=True))
