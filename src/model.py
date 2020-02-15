import tensorflow.keras as keras
from tensorflow.keras import layers

from main import NUM_SAMPLES_PER_BATCH


def create_uncompiled_model():
    model = keras.Sequential()

    model.add(layers.TimeDistributed(layers.Dense(
        units=16, kernel_initializer='random_uniform', bias_initializer='random_uniform', input_shape=(None, 1)
    ), input_shape=(NUM_SAMPLES_PER_BATCH, 1)))

    model.add(layers.SimpleRNN(
        16, kernel_initializer='random_uniform', bias_initializer='random_uniform',
        recurrent_initializer='random_uniform', return_sequences=True
    ))

    model.add(layers.Dense(1, kernel_initializer='random_uniform', bias_initializer='random_uniform'))

    return model


def create_compiled_model():
    model = create_uncompiled_model()
    model.compile(optimizer=keras.optimizers.RMSprop(), loss=keras.losses.MeanSquaredError(), metrics=['mse'])

    print(model.summary())

    return model
