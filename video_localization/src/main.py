import argparse

from data import annotate_dataset, create_video_dataset, show_dataset
from util import test_camera


def do_train_conv_model(args):
    from model.train_conv_model import train_conv_model
    train_conv_model(args)


def parse_args():
    parser = argparse.ArgumentParser(description='Creates a dataset or tests the camera.')
    sub_parsers = parser.add_subparsers()

    # test camera
    test_camera_parser = sub_parsers.add_parser('test-camera', description='Test the connected camera.')
    test_camera_parser.set_defaults(func=test_camera.test_camera)

    # create video dataset
    create_dataset_parser = sub_parsers.add_parser(
        'create-video-dataset', description='Creates a video dataset inside the given database.'
    )
    create_dataset_parser.add_argument(
        'database_directory', metavar='database-directory', type=str, help='The location for the database.'
    )
    create_dataset_parser.set_defaults(func=create_video_dataset.create_video_dataset)

    # annotate dataset
    annotate_dataset_parser = sub_parsers.add_parser('annotate-dataset', description='Annotates a dataset')
    annotate_dataset_parser.add_argument(
        'database_directory', metavar='database-directory', type=str, help='The location of the database'
    )
    annotate_dataset_parser.add_argument(
        '--change', action='store_true', help='If set already present annotations can be changed'
    )
    annotate_dataset_parser.set_defaults(func=annotate_dataset.annotate_dataset)

    # show dataset
    show_dataset_parser = sub_parsers.add_parser('show-dataset', description='Shows the content of a dataset')
    show_dataset_parser.add_argument(
        'database_directory', metavar='database-directory', type=str, help='The location of the database'
    )
    show_dataset_parser.set_defaults(func=show_dataset.show_dataset)

    # train conv model
    train_conv_model_parser = sub_parsers.add_parser(
        'train-conv-model', description='Trains a convolutional model on a given database'
    )
    train_conv_model_parser.add_argument(
        'database_directory', metavar='database-directory', type=str,
        help='The path to the database that is used for training'
    )
    train_conv_model_parser.set_defaults(func=do_train_conv_model)

    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
