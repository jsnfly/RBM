import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout


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
        else:
            hbiases = np.zeros_like(layer.hbiases, dtype=np.float32)
        dbn_weights.append(np.squeeze(hbiases))
    model.set_weights(dbn_weights)

    # compile model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
