import os
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from MNIST_Test.Run_MNIST import load_mnist_data, threshold_data
from evaluate.helper_functions import load_dbn
from evaluate.finetune_dbn import keras_model_from_dbn
from evaluate.feedforward import *
from tensorflow.keras import backend as K
# from pathlib import Path


save_path = os.path.join('Test1', 'finetune/')

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
dropout_rates = [0.0, 0.0, 0.0]
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model = keras_model_from_dbn(dbn, num_classes, dropout_rates, optimizer, use_hbiases=True)
model.summary()

# train model:
batch_size = 128
epochs = 50
history = model.fit(x=x_train[:3000],
                    y=y_train[:3000],
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    verbose=2)

pickle_out = open(save_path + 'finetune_history.pickle', 'wb')
pickle.dump(history.history, pickle_out)
pickle_out.close()

plt.plot(history.history['acc'], label='Train Acc')
plt.plot(history.history['val_acc'], label='Test Acc')
plt.savefig(save_path + 'finettune_acc.png', format='png')
plt.close()
K.clear_session()

# compare with feedforward
finetune_test_acc = history.history['val_acc']

# set up model
num_classes = 10
layer_sizes = [dbn[0].num_vunits]
layer_sizes = layer_sizes + [layer.num_hunits for layer in dbn]
dropout_rates = [0.0, 0.0, 0.0]
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model = feedforward_network(layer_sizes=layer_sizes,
                            dropout_rates=dropout_rates,
                            num_classes=num_classes,
                            optimizer=optimizer,
                            activation='sigmoid')
model.summary()

# train model:
batch_size = 128
epochs = 50
history = model.fit(x=x_train[:3000],
                    y=y_train[:3000],
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    verbose=2)

pickle_out = open(save_path + 'feedforward_history.pickle', 'wb')
pickle.dump(history.history, pickle_out)
pickle_out.close()

plt.plot(history.history['acc'], label='Train Acc')
plt.plot(history.history['val_acc'], label='Test Acc')
plt.savefig(save_path + 'feedforward_acc.png', format='png')
plt.close()

feedforward_test_acc = history.history['val_acc']

plt.plot(finetune_test_acc)
plt.plot(feedforward_test_acc)
plt.show()
