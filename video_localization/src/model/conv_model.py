import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow_core.python.keras import regularizers


def create_uncompiled_conv_model(input_shape):
    """
    Creates a sequential keras model with the given input shape.

    :param input_shape: The expected shape of the input data
    :type input_shape: tuple[int]
    :return: A sequential keras model
    :rtype: keras.Sequential
    """
    model = keras.Sequential()

    model.add(layers.Conv2D(
        filters=32, kernel_size=(3, 3), strides=1, activation='relu', input_shape=input_shape,
        kernel_regularizer=regularizers.l2(0.001)
    ))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(
        filters=32, kernel_size=(3, 3), strides=1, activation='relu',
        kernel_regularizer=regularizers.l2(0.001)
    ))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(
        filters=32, kernel_size=(3, 3), strides=2, activation='relu',
        kernel_regularizer=regularizers.l2(0.001))
    )
    # model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Flatten())
    # model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(2, activation='linear'))

    return model


def create_compiled_conv_model(input_shape):
    """
    Creates a compiled keras convolutional model with the given input shape.
    Uses MeanSquaredError as Loss and Adam Optimizer.

    :param input_shape: The expected shape of the input data
    :type input_shape: tuple[int]
    :return: A sequential keras model
    :rtype: keras.Sequential
    """
    model = create_uncompiled_conv_model(input_shape)

    model.compile(loss=keras.losses.MeanSquaredError(), optimizer=keras.optimizers.Adam())

    return model
