import tensorflow.keras as keras
from tensorflow.keras import layers, regularizers


def create_uncompiled_conv_model(
        input_shape,
        num_filters1=64,
        num_filters2=32,
        num_filters3=16,
        regularization1=0.001,
        regularization2=0.001,
        regularization3=0.001,
        num_nodes_dense1=64,
        regularization_dense1=0.001,
        regularization_dense_bias1=0.005,
        regularization_output_bias=0.005,
):
    """
    Creates a sequential keras model with the given input shape.

    :param input_shape: The expected shape of the input data
    :type input_shape: tuple[int]

    :return: A sequential keras model
    :rtype: keras.Sequential
    """
    model = keras.Sequential()

    model.add(layers.Conv2D(
        filters=num_filters1, kernel_size=(4, 4), strides=1, activation='relu', input_shape=input_shape,
        kernel_regularizer=regularizers.l2(regularization1)
    ))
    # model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(
        filters=num_filters2, kernel_size=(2, 2), strides=1, activation='relu',
        kernel_regularizer=regularizers.l2(regularization2)
    ))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(
        filters=num_filters3, kernel_size=(2, 2), strides=1, activation='relu',
        kernel_regularizer=regularizers.l2(regularization3))
    )
    # model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Flatten())
    model.add(
        layers.Dense(
            num_nodes_dense1,
            activation='relu',
            kernel_regularizer=regularizers.l2(regularization_dense1),
            bias_regularizer=regularizers.l2(regularization_dense_bias1)
        )
    )
    model.add(layers.Dense(2, activation='linear', bias_regularizer=regularizers.l1(regularization_output_bias)))

    return model


def create_compiled_conv_model(input_shape, **hyperparameters):
    """
    Creates a compiled keras convolutional model with the given input shape.
    Uses MeanSquaredError as Loss and Adam Optimizer.

    :param input_shape: The expected shape of the input data
    :type input_shape: tuple[int]
    :param hyperparameters: The hyperparameters, which are forwarded to create_uncompiled_conv_model
    :type hyperparameters: dict[str, Any]
    :return: A sequential keras model
    :rtype: keras.Sequential
    """
    model = create_uncompiled_conv_model(input_shape, **hyperparameters)

    model.compile(
        # loss=keras.losses.MeanSquaredError(),
        loss=keras.losses.MeanAbsoluteError(),
        optimizer=keras.optimizers.Adam(learning_rate=hyperparameters.get('learning_rate', 0.0003)),
        metrics=[keras.metrics.MeanAbsoluteError()]
    )

    return model
