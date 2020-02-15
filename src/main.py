import numpy as np
import model


NUM_BATCHES = 64
NUM_SAMPLES_PER_BATCH = 1024


def create_train_data():
    x_range = np.linspace(0, np.pi * 2, NUM_SAMPLES_PER_BATCH)
    output_sequence: np.ndarray = np.sin(x_range) * 0.8

    data_shape = NUM_BATCHES, NUM_SAMPLES_PER_BATCH

    x_data = np.empty(data_shape)
    y_data = np.empty(data_shape)

    for batch_index in range(NUM_BATCHES):
        random_sequence = np.random.randn(*output_sequence.shape) * 0.07
        input_sequence = output_sequence + random_sequence

        x_data[batch_index] = input_sequence
        y_data[batch_index] = output_sequence

    x_data = x_data.reshape((NUM_BATCHES, NUM_SAMPLES_PER_BATCH, 1))
    y_data = y_data.reshape((NUM_BATCHES, NUM_SAMPLES_PER_BATCH, 1))

    return x_data, y_data


def main():
    x_data, y_data = create_train_data()

    train_model = model.create_compiled_model()

    print('x_data shape: ', x_data.shape)
    print('y_data shape: ', y_data.shape)

    history = train_model.fit(x=x_data, y=y_data, batch_size=1, epochs=3)

    print('history: {}'.format(history))


if __name__ == '__main__':
    main()
