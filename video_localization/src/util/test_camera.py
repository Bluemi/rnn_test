from util.camera import Camera
from util.util import RenderWindow, FPS, KeyCodes


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
        if key == KeyCodes.ESCAPE:
            break
        elif key != -1:
            print('key: {}'.format(key))

        fps.update()
        print(fps.get_fps(), flush=True)

    render_window.close()
    camera.close()
