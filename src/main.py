import numpy as np
import model
import matplotlib.pylab as plt


NUM_BATCHES = 128
NUM_SAMPLES_PER_BATCH = 1024
NOISE_INTENSITY = 0.2


def create_train_data(num_batches):
    x_range = np.linspace(0, np.pi * 2, NUM_SAMPLES_PER_BATCH)

    data_shape = num_batches, NUM_SAMPLES_PER_BATCH

    x_data = np.empty(data_shape)
    y_data = np.empty(data_shape)

    for batch_index in range(num_batches):
        output_sequence: np.ndarray = np.sin(x_range + np.random.rand() * np.pi) * 0.8

        random_sequence = np.random.randn(*output_sequence.shape) * NOISE_INTENSITY
        input_sequence = output_sequence + random_sequence

        x_data[batch_index] = input_sequence
        y_data[batch_index] = output_sequence

    x_data = x_data.reshape((num_batches, NUM_SAMPLES_PER_BATCH, 1))
    y_data = y_data.reshape((num_batches, NUM_SAMPLES_PER_BATCH, 1))

    return x_data, y_data


def plot_results(input_data, result):
    x_data, y_data = input_data

    x_data = x_data.reshape((x_data.shape[0], x_data.shape[1]))
    y_data = y_data.reshape((y_data.shape[0], y_data.shape[1]))
    result = result.reshape((result.shape[0], result.shape[1]))

    x_range = np.linspace(0, np.pi * 2, NUM_SAMPLES_PER_BATCH)

    for batch_index in range(x_data.shape[0]):
        plt.plot(x_range, y_data[batch_index])
        plt.plot(x_range, x_data[batch_index])
        plt.plot(x_range, result[batch_index])
        plt.show()


def main():
    x_train_data, y_train_data = create_train_data(NUM_BATCHES)
    val_data = create_train_data(32)

    train_model = model.create_compiled_model()

    train_model.fit(x=x_train_data, y=y_train_data, batch_size=32, epochs=140, validation_data=val_data)

    test_data = create_train_data(64)

    result = train_model.predict(test_data)

    plot_results(test_data, result)


if __name__ == '__main__':
    main()
