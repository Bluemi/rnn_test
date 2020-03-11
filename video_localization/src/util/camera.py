import cv2
from numpy import ndarray


class Camera:
    def __init__(self, video_capture):
        """
        Creates a new Camera

        :param video_capture: The video capture to take
        :type video_capture: cv2.VideoCapture
        """
        self.video_capture = video_capture

    @staticmethod
    def create():
        """
        Creates a new Camera.

        :return: A new Camera
        :rtype: Camera
        """
        video_capture = cv2.VideoCapture(0)
        return Camera(video_capture)

    def next_frame(self):
        """
        Gets the next frame of the camera.
        The frame will have the shape (height, width, depth).
        The frame has dtype int and all values are between 0 and 255.

        :return: The next frame
        :rtype: ndarray

        :raise ValueError: If the next frame could not be retrieved.
        """
        return_val, frame = self.video_capture.read()

        if not return_val:
            raise ValueError('Failed to get next frame. Return value: {}'.format(return_val))

        return frame

    def close(self):
        if self.video_capture:
            self.video_capture.release()
