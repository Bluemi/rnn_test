import numpy as np
import tensorflow as tf

from train.train_conv_model import IMAGE_SIZE
from util.camera import Camera
from util.images import draw_cross
from util.util import RenderWindow, KeyCodes


MAX_PIXEL_VALUE = 255


def run_model(args):
    model = tf.keras.models.load_model(args.model)

    camera = Camera.create()
    render_window = RenderWindow(args.model)

    print('IMAGE_SIZE: ', IMAGE_SIZE)

    while True:
        frame = camera.next_frame()

        frame = frame.astype(np.float32) / MAX_PIXEL_VALUE
        scaled_frame = tf.image.resize(frame, IMAGE_SIZE)
        scaled_frame = tf.reshape(scaled_frame, (1, *IMAGE_SIZE, 3))

        prediction = model.predict_step(scaled_frame)

        prediction = prediction[0].numpy()

        position = (int(prediction[0]*frame.shape[0]), int(prediction[1] * frame.shape[1]))

        draw_cross(frame, position, size=15)

        key = render_window.show_frame(frame, wait_key_duration=10)
        if key == KeyCodes.ESCAPE:
            break
        elif key != -1:
            print('key: {}'.format(key))

    render_window.close()
    camera.close()
