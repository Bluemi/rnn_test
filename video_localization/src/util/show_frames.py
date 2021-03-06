import numpy as np

from data.data import AnnotatedDataset, VideoDataset
from util.images import draw_cross, get_zoomed_image, translate_position
from util.images.draw_functions import create_draw_addition, dark_version
from util.util import KeyCodes, ACTION_NEXT_KEYS, ACTION_PREVIOUS_KEYS, RenderWindow


DEFAULT_ZOOM_RENDERER_OUTPUT_SIZE = (801, 801)
DEFAULT_KEY_CALLBACK_MOVE_SPEED = 2
DEFAULT_ANNOTATION_COLORS = [np.array([0.0, 0.7, 0.0]), np.array([0.0, 0.0, 0.7]), np.array([0.7, 0.0, 0.0])]
AUTO_PLAY_INTERVAL = 20


class ShowFramesState:
    def __init__(self, num_frames, resolution, wait_key_duration=0):
        self.current_index = 0
        self.wait_key_duration = wait_key_duration
        self.running = True
        self.num_frames = num_frames
        self.render_position = (resolution[0]/2, resolution[1]/2)
        self.zoom = 1.0

    def inc_index(self):
        if self.current_index < self.num_frames - 1:
            self.current_index += 1
        else:
            print('end of video', flush=True)

    def dec_index(self):
        if self.current_index >= 1:
            self.current_index -= 1
        else:
            print('begin of video', flush=True)

    def zoom_in(self, zoom_factor=1.5):
        self.zoom *= zoom_factor

    def zoom_out(self, zoom_factor=1.5):
        self.zoom /= zoom_factor

    def move(self, direction):
        self.render_position = (
            self.render_position[0] + direction[0] / self.zoom,
            self.render_position[1] + direction[1] / self.zoom
        )


def default_key_callback(frames_state, key):
    """
    Changes the frames_state, depending on key.

    :param frames_state: The ShowFramesState object to handle
    :type frames_state: ShowFramesState
    :param key: The pressed key
    :type key: int
    :return: True, if the key was applied otherwise False
    :rtype: bool
    """
    if key in (KeyCodes.ESCAPE, KeyCodes.ENTER):
        frames_state.running = False
    elif key in ACTION_NEXT_KEYS:
        frames_state.inc_index()
    elif key in ACTION_PREVIOUS_KEYS:
        frames_state.dec_index()
    elif key == KeyCodes.H:
        frames_state.move((0, -DEFAULT_KEY_CALLBACK_MOVE_SPEED))
    elif key == KeyCodes.J:
        frames_state.move((DEFAULT_KEY_CALLBACK_MOVE_SPEED, 0))
    elif key == KeyCodes.K:
        frames_state.move((-DEFAULT_KEY_CALLBACK_MOVE_SPEED, 0))
    elif key == KeyCodes.L:
        frames_state.move((0, DEFAULT_KEY_CALLBACK_MOVE_SPEED))
    elif key == KeyCodes.U:
        frames_state.zoom_in()
    elif key == KeyCodes.I:
        frames_state.zoom_out()
    else:
        return False
    return True


class FillAnnotationsKeySupplier:
    def __init__(self, annotations, resolution):
        """
        Creates a new FillAnnotationsKeySupplier.

        :param annotations: The annotations to fill
        :type annotations: np.ndarray
        :param resolution: The video resolution as (height, width, depth). The depth parameter is optional.
        :type resolution: tuple[int, int] or tuple[int, int, int]
        """
        self.annotations = annotations
        self.resolution = resolution[:2]
        self.auto_play = False
        self.auto_play_counter = 0

    def _set_point(self, current_index, render_position):
        annotation_position = (
            render_position[0] / self.resolution[0],
            render_position[1] / self.resolution[1]
        )
        self.annotations[current_index] = annotation_position

    def _annotation_finished(self):
        """
        Returns True, if all annotations are not nan

        :return: True, if all annotations are not nan, otherwise False
        :rtype: bool
        """
        return np.all(np.isfinite(self.annotations))

    def _get_missing_annotation_index(self):
        """
        Returns the first index, where annotations are missing. If no annotations are missing None is returned.

        :return: first index with missing annotations, otherwise None
        :rtype: int or None
        """
        for index, annotation in enumerate(self.annotations):
            if np.all(np.isnan(annotation)):
                return index

        return None

    def _center(self, frames_state):
        """
        Centers the screen to the current annotation.

        :param frames_state: The current frame state
        :type frames_state: ShowFramesState
        """
        current_annotations = self.annotations[frames_state.current_index]
        if np.all(np.isfinite(current_annotations)):
            frames_state.render_position = tuple(current_annotations * self.resolution)

    def _interpolate(self, current_index):
        """
        Autocompletes not annotated frames.

        :param current_index: The current frame index
        :type current_index: int
        """
        start_index = current_index - 1
        while not self._is_annotated(start_index) and start_index >= 0:
            start_index -= 1

        if start_index < 0:
            print('No earlier interpolation frame detected')
            return

        if not self._is_annotated(current_index):
            print('current frame not annotated')
            return

        start_point = self.annotations[start_index]
        end_point = self.annotations[current_index]

        self.annotations[start_index:current_index+1] = np.linspace(start_point, end_point, current_index-start_index+1)

    def _interpolate_forward(self, frames_state):
        """
        Sets the render position to match the interpolation from the last two frames.

        :param frames_state: The current frames state
        :type frames_state: ShowFramesState
        """
        current_index = frames_state.current_index
        if current_index < 2 or self._is_annotated(current_index):
            return
        if not self._is_annotated(current_index-1) or not self._is_annotated(current_index-2):
            return
        diff = self.annotations[current_index-1] - self.annotations[current_index-2]
        frames_state.render_position = diff * self.resolution + frames_state.render_position

    def _is_annotated(self, index):
        """
        Returns whether the given frame is annotated.

        :param index: The index to check
        :type index: int
        :return: True, if the given frame is annotated otherwise False
        :rtype: bool
        """
        return np.all(np.isfinite(self.annotations[index]))

    def _forward(self, frames_state):
        """
        Sets the current point, increases the frames index and interpolates forward.

        :param frames_state: The ShowFramesState object to handle
        :type frames_state: ShowFramesState
        """
        self._set_point(frames_state.current_index, frames_state.render_position)
        print('> {}/{}'.format(frames_state.current_index, frames_state.num_frames))
        frames_state.inc_index()
        self._interpolate_forward(frames_state)

    def __call__(self, frames_state, key):
        """
        Changes the frames_state, depending on key and fills out the given annotations.

        :param frames_state: The ShowFramesState object to handle
        :type frames_state: ShowFramesState
        :param key: The pressed key
        :type key: int
        :return: True, if the key was applied otherwise False
        :rtype: bool
        """
        if key == KeyCodes.SPACE or (key == 0 and self.auto_play):
            self._forward(frames_state)
        elif key == -1:
            if self.auto_play:
                self.auto_play_counter -= 1
                if self.auto_play_counter <= 0:
                    self.auto_play_counter = AUTO_PLAY_INTERVAL
                    self._forward(frames_state)
        elif key == KeyCodes.ENTER:
            self._set_point(frames_state.current_index, frames_state.render_position)
        elif key == KeyCodes.ESCAPE:
            if self._annotation_finished():
                frames_state.running = False
            else:
                missing_index = self._get_missing_annotation_index()
                print('Index {} is not annotated.'.format(missing_index))
        elif key == KeyCodes.W:
            missing_index = self._get_missing_annotation_index()
            if missing_index is not None:
                frames_state.current_index = missing_index
            else:
                print('everything is annotated')
        elif key == KeyCodes.C:
            self._center(frames_state)
        elif key == KeyCodes.T:
            if not self._is_annotated(frames_state.current_index):
                self._set_point(frames_state.current_index, frames_state.render_position)
            self._interpolate(frames_state.current_index)
            frames_state.inc_index()
        elif key == KeyCodes.P:
            self.auto_play = not self.auto_play
        else:
            return default_key_callback(frames_state, key)
        return True


class DefaultMouseSupplier:
    def __init__(self):
        """
        Creates a nwe DefaultMouseSupplier
        """
        self.last_position = None
        self.pressed = False

    def __call__(self, event_type, x, y, _unused1, frames_state):
        """
        Moves the render position depending on mouse movements.

        :param event_type: The mouse event type
        :type event_type: int
        :param x: The x position of the mouse
        :type x: int
        :param y: The y position of the mouse
        :type y: int
        :param frames_state: The ShowFramesStates object to handle
        :type frames_state: ShowFramesState
        """
        if self.last_position is not None and self.pressed:
            diff = (self.last_position[0] - y, self.last_position[1] - x)
            frames_state.move(diff)

        if event_type == 1:
            self.pressed = True
        elif event_type == 4:
            self.pressed = False
        elif event_type == 0:
            self.last_position = (y, x)


def default_frame_callback(frame_state, frames):
    """
    Returns the current frame.

    :param frame_state: The current frame state
    :type frame_state: ShowFramesState
    :param frames: The frames to show
    :type frames: list[np.ndarray] or np.ndarray
    :return: The current frame
    :rtype: np.ndarray
    """
    return frames[frame_state.current_index]


class RenderAnnotationsSupplier:
    def __init__(self, annotations):
        """
        Renders the current frame and a little cross, for the annotations.

        :param annotations: The annotations to draw
        :type annotations: np.ndarray
        """
        self.annotations = annotations

    def __call__(self, frames_state, frames):
        """
        Renders the current frame and a little cross, for the annotations.

        :param frames_state: The current frame state
        :type frames_state: ShowFramesState
        :param frames: The frames to show
        :type frames: list[np.ndarray] or np.ndarray
        :return: The current frame
        :rtype: np.ndarray
        """
        index = frames_state.current_index
        current_frame = frames[index].copy()

        height = current_frame.shape[0]
        width = current_frame.shape[1]

        rel_y, rel_x = self.annotations[index]

        if 0 < rel_y < 1 and 0 < rel_x < 1:
            y = int(rel_y * height)
            x = int(rel_x * width)

            addition_diff = 0.4
            if 0 <= rel_x <= 1 and 0 <= rel_y <= 1 and np.mean(current_frame[x, y]) > 0.5:
                addition_diff = -0.4
            draw_cross(current_frame, (y, x), draw_function=create_draw_addition(addition_diff))

        return current_frame


class ZoomRenderer:
    def __init__(self, output_size=None, enable_cross=False):
        """
        Creates a new ZoomRenderer.

        :param output_size: The size for the output image
        :type output_size: tuple[int, int]
        :param enable_cross: If True, the output image contains a cross in the center
        :type enable_cross: bool
        """
        self.output_size = output_size or DEFAULT_ZOOM_RENDERER_OUTPUT_SIZE
        self.enable_cross = enable_cross

    def __call__(self, frames_state, frames):
        """
        Renders the current frame using zoom and render position.

        :param frames_state: The current frame state
        :type frames_state: ShowFramesState
        :param frames: The frames to show
        :type frames: list[np.ndarray] or np.ndarray
        :return: The current frame
        :rtype: np.ndarray
        """
        current_frame = frames[frames_state.current_index]

        output_image = get_zoomed_image(
            current_frame,
            frames_state.zoom,
            self.output_size,
            frames_state.render_position
        )

        if self.enable_cross:
            cross_position = (self.output_size[0] // 2, self.output_size[1] // 2)
            if np.mean(output_image[cross_position]) > 0.5:
                draw_cross(output_image, cross_position, draw_function=create_draw_addition(-0.4))
            else:
                draw_cross(output_image, cross_position, draw_function=create_draw_addition(0.4))

        return output_image


class ZoomAnnotationsRenderer(ZoomRenderer):
    def __init__(self, annotations, resolution, output_size=None, enable_cross=None, annotation_colors=None):
        """
        Creates a new ZoomAnnotationsRenderer.

        :param annotations: The annotations to render. Can be given as list of annotations or single annotations as
                            np.ndarray
        :type annotations: list[np.ndarray] or np.ndarray
        :param resolution: The resolution of the video as (height, width, depth). The depth parameter is optional.
        :type resolution: tuple[int, int, int]
        :param output_size: The size for the output image
        :type output_size: tuple[int, int]
        :param annotation_colors: The colors to use for the annotations
        :type annotation_colors: list[np.ndarray] or None
        """
        super().__init__(output_size, enable_cross)
        if isinstance(annotations, list):
            self.annotations = annotations
        elif isinstance(annotations, np.ndarray):
            self.annotations = [annotations]
        self.resolution = resolution[:2]
        self.annotation_colors = annotation_colors or DEFAULT_ANNOTATION_COLORS

    def __call__(self, frames_state, frames):
        """
        Renders the current frame using zoom and render position. Draws a cross at the annotation position.

        :param frames_state: The current frame state
        :type frames_state: ShowFramesState
        :param frames: The frames to show
        :type frames: list[np.ndarray] or np.ndarray
        :return: The current frame
        :rtype: np.ndarray
        """
        output_image = super().__call__(frames_state, frames)

        for annotations, annotation_color in zip(self.annotations, self.annotation_colors):
            current_annotations = annotations[frames_state.current_index]

            if not (np.isnan(current_annotations[0]) or np.isnan(current_annotations[1])):
                draw_position = translate_position(
                    (current_annotations[0] * self.resolution[0], current_annotations[1] * self.resolution[1]),
                    self.output_size,
                    frames_state.render_position,
                    frames_state.zoom
                )
                draw_position = tuple(int(x) for x in draw_position)
                used_color = annotation_color
                if 0 <= draw_position[0] < output_image.shape[0] and 0 <= draw_position[1] < output_image.shape[1]:
                    if not np.mean(output_image[draw_position]) < 0.5:
                        used_color = dark_version(annotation_color)

                draw_cross(
                    output_image,
                    draw_position,
                    draw_function=create_draw_addition(used_color)
                )

        return output_image


def show_frames(
        data_source,
        window_title='frames',
        key_callback=None,
        mouse_callback=None,
        render_callback=None,
        window_position=None
):
    """
    Shows the given frames

    :param data_source: The source of the frames to show. Can be a numpy.ndarray, list[numpy.ndarray], Dataset or
                        VideoDataset
    :type data_source: list[np.ndarray] or np.ndarray or AnnotatedDataset or VideoDataset
    :param window_title: The title of the window
    :type window_title: str
    :param key_callback: Callback for every keystroke. Should take the current frame_state object, as well as the
                         pressed key
    :type key_callback: Callable or None
    :param mouse_callback: Callback for mouse movements. Should take the current frame_state object, as well as the
    :type mouse_callback: Callable or None
    :param render_callback: Callable that returns the ndarray to render
    :type render_callback: Callable or None
    :param window_position: The initial position of the window
    :type window_position: tuple[int, int] or None

    :raise TypeError: If data_source has invalid type
    """
    if isinstance(data_source, np.ndarray):
        frames = data_source
    elif isinstance(data_source, list):
        for data_source_element in data_source:
            if not isinstance(data_source_element, np.ndarray):
                raise TypeError('Cant show frames for list of type "{}"'.format(type(data_source_element).__name__))
        frames = data_source
    elif isinstance(data_source, AnnotatedDataset):
        frames = data_source.video_data
        if render_callback is None:
            render_callback = RenderAnnotationsSupplier(data_source.annotation_data)
    elif isinstance(data_source, VideoDataset):
        frames = data_source.video_data
    else:
        raise TypeError('Cant show frames for data source of type "{}"'.format(type(data_source).__name__))

    _show_frames_impl(frames, window_title, key_callback, mouse_callback, render_callback, window_position)


def _show_frames_impl(
        frames,
        window_title='frames',
        key_callback=None,
        mouse_callback=None,
        frame_callback=None,
        window_position=None
):
    """
    Shows the given frames

    :param frames: The frames to show
    :type frames: list[np.ndarray] or np.ndarray
    :param window_title: The title of the window
    :type window_title: str
    :param key_callback: Callback for every keystroke. Should take the current frame_state object, as well as the
                         pressed key
    :type key_callback: Callable or None
    :param mouse_callback: Callback for mouse movements. Should take the current frame_state object, as well as the
    :type mouse_callback: Callable or None
    :param frame_callback: Callable that returns the ndarray to render
    :type frame_callback: Callable or None
    :param window_position: The initial window position given as tuple (y, x)
    :type window_position: tuple[int, int] or None

    :return: Returns the control in its end state
    :rtype: ShowFramesState
    """
    frames_state = ShowFramesState(len(frames), frames[0].shape, wait_key_duration=10)

    if key_callback is None:
        key_callback = default_key_callback

    if mouse_callback is None:
        mouse_callback = DefaultMouseSupplier()

    if frame_callback is None:
        frame_callback = default_frame_callback

    window = RenderWindow(window_title, position=window_position)
    if mouse_callback is not None:
        window.set_mouse_callback(mouse_callback, frames_state)

    while frames_state.running:
        key = window.show_frame(frame_callback(frames_state, frames), wait_key_duration=frames_state.wait_key_duration)
        key_callback(frames_state, key)

    window.close()

    return frames_state
