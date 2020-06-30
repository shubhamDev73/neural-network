def load_dataset(dataset):
    # loading
    (train_images, train_labels), (test_images, test_labels) = dataset.load_data()

    # converting intensity to 0..1 value
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    return (train_images, train_labels), (test_images, test_labels)
