import argparse

from data import annotate_dataset, create_video_dataset, show_dataset
from data.import_video import import_video
from util import test_camera


def do_train_conv_model(args):
    from train.train_conv_model import train_conv_model
    train_conv_model(args)


def do_run_model(args):
    from train.run_model import run_model
    run_model(args)


def do_eval_model(args):
    from train.evaluate_model import evaluate_model
    evaluate_model(args)


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
        'train_data', metavar='train-data', type=str,
        help='The path to the database that is used for training'
    )
    train_conv_model_parser.add_argument(
        '--show', action='store_true',
        help='If set the predictions of the model are shown using the validation data, if present'
    )
    train_conv_model_parser.set_defaults(func=do_train_conv_model)

    # import video
    import_video_parser = sub_parsers.add_parser(
        'import-video', description='Imports a video file. The result is a VideoDataset'
    )
    import_video_parser.add_argument(
        'database_directory', metavar='database-directory', type=str, help='The location of the database'
    )
    import_video_parser.add_argument(
        'video_file', metavar='video-file', type=str, help='The video file to convert'
    )
    import_video_parser.add_argument(
        '--flip', action='store_true', help='If set the video is flipped horizontally'
    )
    import_video_parser.set_defaults(func=import_video)

    # run model
    run_model_parser = sub_parsers.add_parser(
        'run-model', description='Runs the given model live using a webcam'
    )
    run_model_parser.add_argument('model', type=str, help='The model to run')
    run_model_parser.set_defaults(func=do_run_model)

    # evaluate model
    evaluate_model_parser = sub_parsers.add_parser('eval-model', description='Evaluates the given model')
    evaluate_model_parser.add_argument(
        'train_data', metavar='train-data', type=str,
        help='The path to the database that is used for training'
    )
    evaluate_model_parser.add_argument('model', type=str, help='The model to run')
    evaluate_model_parser.add_argument(
        '--show', action='store_true',
        help='If set the predictions of the model are shown'
    )
    evaluate_model_parser.set_defaults(func=do_eval_model)

    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
