import argparse
import cv2
import numpy as np

from data.create_video_dataset import dump_video_dataset, get_dataset_directory, choose_frames
from data.data import DataError

IMPORT_RESOLUTION = (640, 480)
FRAME_DIFF_LIMIT = 500000


def import_video(args):
    """
    Imports a video and converts it into a VideoDataset.

    :param args: The arguments for this import
    :type args: argparse.Namespace
    """
    video_file = args.video_file

    try:
        vid_file = cv2.VideoCapture(video_file)
    except Exception as e:
        raise DataError('Could not open video file.\n{}'.format(str(e)))

    if not vid_file.isOpened():
        raise DataError('Video file is not opened')

    frames = []
    ret, frame = vid_file.read()
    while ret:
        if not frames:
            frames.append(frame)
        else:
            diff_frame = np.maximum(frames[-1], frame) - np.minimum(frames[-1], frame)
            if np.sum(diff_frame) > FRAME_DIFF_LIMIT:
                frames.append(frame)
        ret, frame = vid_file.read()

    result_frames = []
    for index, frame in enumerate(frames):
        flipped_frame = frame
        if args.flip:
            flipped_frame = cv2.flip(frame, 1)
        resized_frame = cv2.resize(flipped_frame, IMPORT_RESOLUTION, interpolation=cv2.INTER_NEAREST)
        result_frames.append(resized_frame)

    chosen_frames = choose_frames(result_frames)

    dump_video_dataset(np.array(chosen_frames), get_dataset_directory(args.database_directory))
