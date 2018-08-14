import numpy as np
import pickle
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Dropout
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras import backend as K
from Run_MNIST import load_mnist_data
from plot_features import load_dbn
import tensorflow as tf
from pathlib import Path
import os
from RBM import RBM
import matplotlib.pyplot as plt


def finetune_dbn(dbn, train_data, train_labels, test_data, test_labels,
                 optimizer='ADAM', epochs=10, batch_size=128,
                 dropout_rate=0):
    # Build feed-forward net from dbn:
    a = Input(shape=(train_data.shape[1],))
    layer = dbn['layer_0']
    if layer.layer_type == 'gr' or 'cr':
        activation = 'relu'
    else:
        activation = 'sigmoid'
    x = Dense(units=layer.num_hunits, activation=activation, use_bias=True)(a)
    if dropout_rate != 0:
        x = Dropout(dropout_rate)(x)
    for li in range(1, len(dbn)):
        layer = dbn['layer_{}'.format(li)]
        if layer.layer_type == 'gr' or 'cr':
            activation = 'relu'
        else:
            activation = 'sigmoid'
        if layer.layer_type == 'cb' and dbn['layer_{}'.format(li - 1)] == 'gr':
            x = tf.nn.sigmoid(x)
        x = Dense(units=layer.num_hunits, activation=activation, use_bias=True)(x)
        if dropout_rate != 0:
            x = Dropout(dropout_rate)(x)
    o = Dense(units=train_labels.shape[1], activation='softmax')(x)
    weights_and_biases = []
    for li in range(len(dbn)):
        weights = dbn['layer_{}'.format(li)].weights
        weights_and_biases.append(weights)
        biases = np.squeeze(dbn['layer_{}'.format(li)].hbiases)
        weights_and_biases.append(biases)
    model = Model(inputs=a, outputs=o)
    model.set_weights(weights_and_biases)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Train model:
    history = model.fit(x=train_data, y=train_labels, batch_size=batch_size, epochs=epochs,
                        validation_data=(test_data, test_labels))
    K.clear_session()
    return history


# load MNIST data
(x_train, y_train), (x_test, y_test) = load_mnist_data()

# make one hot
y_train = tf.one_hot(y_train, depth=10)
y_test = tf.one_hot(y_test, depth=10)

# load dbn for finetuning
dbn_path = Path('GR_MNIST_512_256_128_64/dbn.pickle')
dbn_path = os.fspath(dbn_path)
dbn = load_dbn(dbn_path)

# print weight absolute values
for li in range(len(dbn)):
    print(np.mean(np.abs(dbn['layer_{}'.format(li)].weights)))

# # scale weights
# dbn['layer_0'].weights = dbn['layer_0'].weights*12
# dbn['layer_1'].weights = dbn['layer_1'].weights*5
# dbn['layer_2'].weights = dbn['layer_2'].weights*1.5

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

history = finetune_dbn(dbn,
                       x_train[:int(0.05*x_train.shape[0]), :],
                       y_train[:int(0.05*y_train.shape[0]), :],
                       x_test,
                       y_test,
                       optimizer, 100, 128, dropout_rate=0.4)

finetune_acc = history.history['val_acc']
plt.plot(finetune_acc)
plt.show()
