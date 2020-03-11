import cv2
import numpy as np


def is_in_range(frame, position):
    """
    Checks whether the given position is inside the frame

    :param frame: The frame to check against. Should have dtype float with values between 0 and 1
    :type frame: np.ndarray
    :param position: The position in absolute coordinates as [y, x]
    :type position: tuple[int, int]
    """
    return 0 <= position[0] < frame.shape[0] and 0 <= position[1] < frame.shape[1]


def draw_point(frame, position, draw_function=None):
    """
    Draws a point into the given frame at the given position.

    :param frame: The frame to draw to. Should have dtype float with values between 0 and 1
    :type frame: np.ndarray
    :param position: The position in absolute coordinates as [y, x]
    :type position: tuple[int, int]
    :param draw_function: The function to calculate the next color. Should accept the old pixel color and returns the
                          new pixel color. Defaults to drawing white.
    :type draw_function: Callable[[float], float] or None
    """
    if is_in_range(frame, position):
        if draw_function is None:
            frame[position] = np.array([1, 1, 1])
        else:
            frame[position] = draw_function(frame[position])


def draw_cross(frame, position, size=3, draw_function=None):
    """
    Draws a cross into the given frame at the given position.

    :param frame: The frame to draw to
    :type frame: np.ndarray
    :param position: The position in absolute coordinates
    :type position: tuple[int, int]
    :param size: The size of the cross. The cross will be size*2 + 1 pixels wide and high
    :param draw_function: The function to calculate the next color. Should accept the old pixel color and returns the
                          new pixel color. Defaults to drawing white.
    :type draw_function: Callable[[float], float] or None
    """
    for y in range(size*2+1):
        pos = (position[0]-size+y, position[1])
        draw_point(frame, pos, draw_function)

    for x in range(size*2 + 1):
        pos = (position[0], position[1]-size+x)
        if x != size:
            draw_point(frame, pos, draw_function)


def get_sub_image(image, position, size):
    """
    Returns a subset of the given image. Parts that which are out of bounds of the source image are filled with zeros.
    The returned sub image will never share memory with the source image.

    :param image: The image to get the sub_image from
    :type image: np.ndarray
    :param position: The left top position of the sub image in the source image, given as tuple (y, x)
    :type position: tuple[int, int]
    :param size: The (height, width) tuple defining the size of the sub image. Should always be positive
    :type size: tuple[int, int]
    :return: the defined sub image
    :rtype: np.ndarray
    """
    if len(image.shape) == 2:
        image_copy_shape = size
    elif len(image.shape) == 3:
        image_copy_shape = (*size, image.shape[2])
    else:
        raise ValueError('Shape of source image should have len 2 or 3')

    image_copy = np.zeros(image_copy_shape, dtype=np.float)
    sub_image = image[
        np.maximum(position[0], 0):np.maximum(position[0]+size[0], 0),
        np.maximum(position[1], 0):np.maximum(position[1]+size[1], 0)
    ]

    if position[0] >= 0:
        y_offset = 0
    else:
        y_offset = -position[0]

    if position[1] >= 0:
        x_offset = 0
    else:
        x_offset = -position[1]

    image_copy[y_offset:y_offset + sub_image.shape[0], x_offset:x_offset + sub_image.shape[1]] = sub_image

    return image_copy


def get_zoomed_image(source_image, zoom, output_size, render_position):
    """
    Extracts an sub image from the given source image.
    The output image will be zoomed in by zoom, have the given output size and will be centered around the given render
    position relative to the source image.

    :param source_image: The source image to extract from
    :type source_image: np.ndarray
    :param zoom: The zoom factor (1 == no zoom, 2 == size doubled)
    :type zoom: float
    :param output_size: The size of the output image (height, width).
    :type output_size: tuple[int, int]
    :param render_position: The position in the source image (y, x)
    :type render_position: tuple[int, int]
    :return: The extracted sub image
    :rtype: np.ndarray
    """
    precise_sub_image_size = (
        output_size[0] / zoom,
        output_size[1] / zoom
    )
    rounded_sub_image_size = tuple(int(x) + 1 for x in precise_sub_image_size)

    precise_sub_image_position = (
        render_position[0] - (precise_sub_image_size[0] / 2),
        render_position[1] - (precise_sub_image_size[1] / 2)
    )
    rounded_sub_image_position = tuple(int(x) + 1 for x in precise_sub_image_position)

    tmp_sub_image = get_sub_image(source_image, rounded_sub_image_position, rounded_sub_image_size)

    tmp_sub_image_scaled_size = tuple(int(x * zoom) for x in rounded_sub_image_size)

    scaled_tmp_sub_image = cv2.resize(tmp_sub_image, tmp_sub_image_scaled_size, cv2.INTER_NEAREST)

    border_position = (
        int((precise_sub_image_position[0] - rounded_sub_image_position[0]) * zoom),
        int((precise_sub_image_position[1] - rounded_sub_image_position[1]) * zoom),
    )

    output_image = get_sub_image(scaled_tmp_sub_image, border_position, output_size)

    return output_image


def translate_position(source_position, output_size, render_position, zoom=1.0):
    """
    Translates the given position into a position relative to the output image.

    :param source_position: The position in the original image
    :type source_position: tuple[float, float]
    :param output_size: The size of the output image
    :type output_size: tuple[int, int]
    :param render_position: The render position relative to the source image
    :type render_position: tuple[float, float]
    :param zoom: The zoom factor
    :type zoom: float
    :return: The translated position
    :rtype: tuple[float, float]
    """
    precise_sub_image_size = (
        output_size[0] / zoom,
        output_size[1] / zoom
    )
    precise_sub_image_position = (
        render_position[0] - (precise_sub_image_size[0] / 2),
        render_position[1] - (precise_sub_image_size[1] / 2)
    )
    offset = (
        (source_position[0] - precise_sub_image_position[0]) * zoom,
        (source_position[1] - precise_sub_image_position[1]) * zoom
    )
    return offset
