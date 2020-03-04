import cv2

from camera import Camera
from util import FPS, RenderWindow


def main():
    camera = Camera.create()
    render_window = RenderWindow('preview')

    fps = FPS()

    while True:
        frame = camera.next_frame()

        key = render_window.show_frame(frame)
        if key == 27:  # exit on ESC
            break

        fps.update()
        print(fps.get_fps(), flush=True)

    render_window.close()
    camera.close()


if __name__ == '__main__':
    main()
