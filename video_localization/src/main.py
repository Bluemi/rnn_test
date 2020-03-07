import argparse

from exec import annotate_dataset, create_video_dataset, show_dataset, test_camera


def parse_args():
    parser = argparse.ArgumentParser(description='Creates a dataset or tests the camera.')
    sub_parsers = parser.add_subparsers()

    # test camera
    test_camera_parser = sub_parsers.add_parser('test-camera', description='Test the connected camera.')
    test_camera_parser.set_defaults(func=test_camera)

    # create video dataset
    create_dataset_parser = sub_parsers.add_parser(
        'create-video-dataset', description='Creates a video dataset inside the given database.'
    )
    create_dataset_parser.add_argument(
        'database_directory', metavar='database-directory', type=str, help='The location for the database.'
    )
    create_dataset_parser.set_defaults(func=create_video_dataset)

    # annotate dataset
    annotate_dataset_parser = sub_parsers.add_parser('annotate-dataset', description='Annotates a dataset')
    annotate_dataset_parser.add_argument(
        'database_directory', metavar='database-directory', type=str, help='The location of the database'
    )
    annotate_dataset_parser.set_defaults(func=annotate_dataset)

    # show dataset
    show_dataset_parser = sub_parsers.add_parser('show-dataset', description='Shows the content of a dataset')
    show_dataset_parser.add_argument(
        'database_directory', metavar='database-directory', type=str, help='The location of the database'
    )
    show_dataset_parser.set_defaults(func=show_dataset)

    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
