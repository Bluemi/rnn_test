import data
import conv_model


def main():
    train_dataset = data.ImageLocalizationDataset.create(20000)
    train_dataset.show()

    model = conv_model.create_compiled_model(train_dataset.get_shape())

    # train
    model.fit(x=train_dataset.get_x_reshaped(), y=train_dataset.y_data, batch_size=5, epochs=3, verbose=True)

    # test
    test_dataset = data.ImageLocalizationDataset.create(20)
    test_results = model.predict(x=test_dataset.get_x_reshaped())

    test_dataset.draw_results(test_results)
    test_dataset.show(3.0)


if __name__ == '__main__':
    main()
