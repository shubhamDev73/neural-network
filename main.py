from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np

import model, data

if __name__ == "__main__":
    # loading dataset
    training_data, testing_data = data.load_dataset(fashion_mnist)

    # creating model
    neural_network = model.create_model() # new model
    # neural_network = model.create_model("data.h5") # loading existing model weights

    # training
    test_acc = model.train(neural_network, training_data, testing_data)
    print("Test accuracy:", test_acc)

    # saving weights to file
    model.save(neural_network)

    # making predictions
    prediction_model = model.create_prediction_model(neural_network)
    predictions = model.predict(prediction_model, testing_data[0])

    # showing images and prediction
    names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    for index, image in enumerate(testing_data[0]):
        # show image
        plt.figure()
        plt.imshow(image, cmap=plt.cm.binary)
        plt.show()

        # prediction
        print("Prediction:", names[np.argmax(predictions[index])], "\tActual:", names[testing_data[1][index]])
