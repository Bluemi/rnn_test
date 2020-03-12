import numpy as np

from data.data import DatasetPlaceholder, chose_dataset_placeholder, VideoDataset, Dataset
from util.show_frames import show_frames, ZoomAnnotationsRenderer, FillAnnotationsKeySupplier


def annotate_dataset(args):
    dataset_placeholders = DatasetPlaceholder.list_database(args.database_directory)

    if args.change:
        dataset_placeholders = list(filter(DatasetPlaceholder.is_full_dataset, dataset_placeholders))
    else:
        dataset_placeholders = list(filter(DatasetPlaceholder.is_video_dataset, dataset_placeholders))

    dataset_placeholder = chose_dataset_placeholder(dataset_placeholders)

    source_dataset = dataset_placeholder.load()

    annotations = annotate_frames(source_dataset)

    np.save(dataset_placeholder.get_annotations_path(), annotations)


def annotate_frames(dataset):
    """
    Returns the full dataset.

    :param dataset: The dataset to annotate
    :type dataset: VideoDataset or Dataset
    :return: The annotations made
    :rtype: np.ndarray
    """
    frames = dataset.video_data

    if dataset.is_video_dataset():
        annotations = np.zeros((len(frames), 2), dtype=np.float) + np.nan
    elif dataset.is_full_dataset():
        annotations = dataset.annotation_data
    else:
        raise TypeError('dataset is not of type VideoDataset nor Dataset')

    render_supplier = ZoomAnnotationsRenderer(annotations, dataset.get_resolution(), enable_cross=True)
    key_supplier = FillAnnotationsKeySupplier(annotations, dataset.get_resolution())

    show_frames(
        dataset.video_data,
        'annotate dataset',
        render_callback=render_supplier,
        key_callback=key_supplier,
        window_position=(200, 200)
    )

    return annotations
