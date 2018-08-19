import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from Run_MNIST import load_mnist_data, threshold_data
from helper_functions import load_dbn
from pathlib import Path


def keras_model_from_dbn(dbn, num_classes, dropout_rates, optimizer, use_hbiases=True):
    """
    create a keras feedforawrd network from a dbn
    :param dbn: Deep belief network (list of RBM instances)
    :param num_classes: number of different classes
    :param dropout_rates: dropout rates for each layer (list of floats between 0 and 1)
    :param optimizer: which optimizer to use for the fine-tuning
    :param use_hbiases: whether to use the hidden biases from pre-training
    :return: model instance
    """
    layer = dbn[0]
    # build feed forward network

    # input layer
    a = Input(shape=(layer.num_vunits,))
    if layer.layer_type == 'gr' or layer.layer_type == 'cr':
        activation = 'relu'
    else:
        activation = 'sigmoid'
    x = Dense(units=layer.num_hunits, activation=activation, use_bias=use_hbiases)(a)
    if dropout_rates[0] != 0:
        x = Dropout(dropout_rates[0])(x)

    # higher layers
    for layer_index in range(1, len(dbn)):
        layer = dbn[layer_index]
        if layer.layer_type == 'gr' or layer.layer_type == 'cr':
            activation = 'relu'
        else:
            activation = 'sigmoid'
        x = Dense(units=layer.num_hunits, activation=activation, use_bias=use_hbiases)(x)
        if dropout_rates[layer_index] != 0:
            x = Dropout(dropout_rates[layer_index])(x)

    # softmax output layer
    o = Dense(units=num_classes, activation='softmax')(x)

    # create model
    model = Model(inputs=a, outputs=o)

    # set model weights (and biases if bool is True) to dbn parameters
    dbn_weights = []
    for layer_index in range(len(dbn)):
        layer = dbn[layer_index]

        weights = layer.weights
        dbn_weights.append(weights)

        if use_hbiases is True:
            hbiases = layer.hbiases
            dbn_weights.append(np.squeeze(hbiases))
    model.set_weights(dbn_weights)

    # compile model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def main():
    # load dbn
    dbn_path = 'Test1/'
    dbn = load_dbn(dbn_path + 'dbn.pickle')

    # load data
    (x_train, y_train), (x_test, y_test) = load_mnist_data()

    # threshold data
    x_train = threshold_data(x_train, 0.5 * 255)
    x_test = threshold_data(x_test, 0.5 * 255)

    # make one hot labels
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # set up model
    num_classes = 10
    dropout_rates = [0.5, 0.5, 0.5]
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model = keras_model_from_dbn(dbn, num_classes, dropout_rates, optimizer, use_hbiases=True)
    model.summary()

    # train model:
    batch_size = 128
    epochs = 50
    history = model.fit(x=x_train,
                        y=y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        verbose=2)

    plt.plot(history.history['acc'], label='Train Acc')
    plt.plot(history.history['val_acc'], label='Test Acc')
    plt.show()


if __name__ == '__main__':
    main()

