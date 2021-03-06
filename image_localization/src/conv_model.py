import tensorflow.keras as keras
from tensorflow.keras import layers


def create_uncompiled_model(input_shape):
    model = keras.Sequential()

    model.add(layers.Conv2D(
        filters=32, kernel_size=(3, 3), strides=2, activation='relu', input_shape=input_shape
    ))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=2, activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(2, activation='linear'))

    return model


def create_compiled_model(input_shape):
    model = create_uncompiled_model(input_shape)

    model.compile(loss=keras.losses.MeanSquaredError(), optimizer=keras.optimizers.Adam())

    return model
