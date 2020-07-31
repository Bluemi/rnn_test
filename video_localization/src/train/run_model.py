import tensorflow as tf

from data.preprocessing import scale_to
from train.train_conv_model import IMAGE_SIZE
from util.camera import Camera
from util.util import RenderWindow, KeyCodes


def run_model(args):
    model = tf.keras.models.load_model(args.model)

    camera = Camera.create()
    render_window = RenderWindow(args.model)

    print('IMAGE_SIZE: ', IMAGE_SIZE)

    while True:
        frame = camera.next_frame()

        scaled_frame = tf.image.resize(frame, IMAGE_SIZE)
        scaled_frame = tf.reshape(scaled_frame, (1, *IMAGE_SIZE, 3))

        prediction = model.predict_step(scaled_frame)

        key = render_window.show_frame(frame, wait_key_duration=10)
        if key == KeyCodes.ESCAPE:
            break
        elif key != -1:
            print('key: {}'.format(key))

    render_window.close()
    camera.close()
