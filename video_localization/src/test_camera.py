from camera import Camera
from util import RenderWindow, FPS, ESCAPE_KEY


def test_camera(_args):
    """
    Opens a window and shows the video stream of the connected camera while printing fps information.

    :param _args: unused
    """
    del _args

    camera = Camera.create()
    render_window = RenderWindow('preview')

    fps = FPS()

    while True:
        frame = camera.next_frame()

        key = render_window.show_frame(frame, wait_key_duration=10)
        if key == ESCAPE_KEY:
            break

        fps.update()
        print(fps.get_fps(), flush=True)

    render_window.close()
    camera.close()


