import cv2


def no_preprocessing(video, anno):
    return video, anno


def scale_to(size):
    """
    Creates a preprocessing function that scaled the images to the given size

    :param size: The size after the preprocessing given as tuple (height, width)
    :type size: tuple[int, int]
    :return: Returns a preprocessing function that scales the image to the given size
    :rtype: Callable[[numpy.ndarray, numpy.ndarray], tuple[numpy.ndarray, numpy.ndarray]]
    """
    def _scale(image_data, annotation_data):
        return cv2.resize(image_data, (size[1], size[0])), annotation_data

    return _scale
