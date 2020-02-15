import numpy as np


def create_train_data():
    x_range = np.linspace(0, np.pi * 2, 300)
    output_sequence: np.ndarray = np.sin(x_range) * 0.8

    random_sequence = np.random.randn(*output_sequence.shape) * 0.07

    input_sequence = output_sequence + random_sequence

    return input_sequence, output_sequence


def main():
    train_data = create_train_data()


if __name__ == '__main__':
    main()
