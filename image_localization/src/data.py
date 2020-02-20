import cv2
import numpy as np


IMAGE_SIZE = np.array([64, 64])


def paint_rectangle(image, absolute_position, size=(0, 0), color=1):
    """
    Creates a white rectangle in the given image.

    :param image: The image to paint to
    :type image: np.ndarray
    :param absolute_position: The position where to paint the rectangle
    :type absolute_position: np.ndarray
    :param size: The size of the rectangle as tuple (x_size, y_size). If size == (0, 0) only one pixel is drawn.
                 If size==(1, 1) one pixel is added top, bottom, left and right.
    :param color: The color of the rectangle
    :type color: float
    :return:
    """
    image[
        max(absolute_position[0]-size[0], 0):min(absolute_position[0]+size[0]+1, image.shape[0]),
        max(absolute_position[1]-size[1], 0):min(absolute_position[1]+size[1]+1, image.shape[1])
    ] = color


class ImageLocalizationDataset:
    def __init__(self, x_data, y_data):
        """
        Creates a new Image Localization Dataset.

        :param x_data: A numpy array with the shape (num_samples, image_height, image_width). Every sample is an image
                       containing a white rectangle.
        :type x_data: np.ndarray
        :param y_data: A numpy array with the shape (num_samples, 2). Every sample contains [y, x] position of the
                       rectangle in the image
        :type y_data: np.ndarray
        """
        self.x_data = x_data
        self.y_data = y_data

    @staticmethod
    def create(num_samples):
        """
        Creates a new Image Localization Dataset.

        :param num_samples: The number of samples contained
        :type num_samples: int
        :return: A new ImageLocalizationDataset
        :rtype: ImageLocalizationDataset
        """
        x_data = np.zeros((num_samples, *IMAGE_SIZE))
        y_data = np.zeros((num_samples, 2))

        for x_sample, y_sample in zip(x_data, y_data):
            np.copyto(y_sample, np.random.rand(2)*0.8 + 0.1)
            absolute_position = ImageLocalizationDataset.relative_image_position_to_absolute(y_sample)
            paint_rectangle(x_sample, absolute_position, size=(1, 1))

        return ImageLocalizationDataset(x_data, y_data)

    def __iter__(self):
        return zip(self.x_data, self.y_data)

    @staticmethod
    def relative_image_position_to_absolute(relative_position):
        return (relative_position * IMAGE_SIZE).astype(int)

    def show(self, scale=1.0):
        """
        Shows a series of images describing the dataset
        """
        for x_sample, y_sample in self:
            cv2.imshow(
                f'x={y_sample[1]:2.2} y={y_sample[0]:2.2}',
                cv2.resize(x_sample.reshape(IMAGE_SIZE), (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            )
            if cv2.waitKey() == 27:
                cv2.destroyAllWindows()
                break

    def get_x_reshaped(self):
        return self.x_data.reshape((*self.x_data.shape, 1))

    def draw_results(self, results):
        """
        Draws the given results into the dataset.

        :param results: The predictions to draw to the dataset. Should be an ndarray with shape (num_samples, 2).
        :type results: np.ndarray
        """
        for x_sample, result in zip(self.x_data, results):
            absolute_image_position = ImageLocalizationDataset.relative_image_position_to_absolute(result)
            paint_rectangle(x_sample, absolute_image_position, color=0.5)

    # noinspection PyMethodMayBeStatic
    def get_shape(self):
        return (*IMAGE_SIZE, 1)
