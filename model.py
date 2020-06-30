from tensorflow import keras

def create_model(file=None):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10),
    ])
    model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    if file:
        model.load_weights(file)

    return model

def create_prediction_model(model):
    return keras.Sequential([model, keras.layers.Softmax()])

def save(model, file="data.h5"):
    model.save_weights(file)

def train(model, training_data, testing_data):
    # training
    model.fit(training_data[0], training_data[1], epochs=10)

    # testing
    test_loss, test_acc = model.evaluate(testing_data[0], testing_data[1], verbose=2)
    return test_acc

def predict(prediction_model, images):
    return prediction_model(images)
