import tensorflow.keras as keras
from tensorflow.keras import layers

from main import NUM_SAMPLES_PER_BATCH


def create_uncompiled_model():
    model = keras.Sequential()

    model.add(layers.TimeDistributed(layers.Dense(
        units=32, kernel_initializer='random_uniform', bias_initializer='random_uniform', input_shape=(None, 1)
    ), input_shape=(NUM_SAMPLES_PER_BATCH, 1)))

    model.add(layers.LSTM(
        32, kernel_initializer='random_uniform', bias_initializer='random_uniform',
        recurrent_initializer='random_uniform', return_sequences=True
    ))

    model.add(layers.Dense(1, kernel_initializer='random_uniform', bias_initializer='random_uniform'))

    return model


def create_compiled_model():
    model = create_uncompiled_model()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=keras.losses.MeanSquaredError())

    print(model.summary())

    return model
