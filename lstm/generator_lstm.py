import sys
sys.path.append("..")

import os
import pickle
import time
from evaluate.helper_functions import layerwise_activations
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Input, Bidirectional
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from lstm.lstm_helper_functions import *

# set paramters
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

NUM_TIME_STEPS = 20
BIDIRECTIONAL = True
LSTM_SIZE = 64
BATCH_SIZE = 64
EPOCHS = 20
STEPS_PER_EPOCH = 25
VALIDATION_STEPS_PER_EPOCH = 10
LEARNING_RATE = 0.001
FORWARD_DROPOUT = 0.5
RECURRENT_DROPOUT = 0.0
SAVE_PATH = "/home/jonas/Desktop/testing/convnet/test1/generator_bidirectional/"
SAVE_NAME = f"{NUM_TIME_STEPS}length_{LSTM_SIZE}size_{int(time.time())}"

# set FEATURE_MODEL to None if no keras model is used
FEATURE_MODEL = "/home/jonas/Desktop/testing/convnet/test1/Model.hdf5"
# LAYER_NAME can be obtained from calling model.summary()
LAYER_NAME = "global_average_pooling1d"
CONV_NET = True

# set DBN_MODEL to None if no dbn is used
DBN_MODEL = None
# LAYER_INDEX is only relevant for DBN models
LAYER_INDEX = -1  # -1 for last layer

if FEATURE_MODEL is not None and DBN_MODEL is not None:
    raise AttributeError("Keras model and DBN model given, set one or both to None!")

# get training and validation files
LOAD_PATH = "/home/jonas/Desktop/testing/raw_data_30s_intervals/"
KEYS = ["sample", "one_hot_label"]
DATA_TYPES = ["float32", "int32"]

training_files = []
validation_files = []

# only works if there is now dataset number higher than 50!
for file in os.listdir(LOAD_PATH):
    if "dataset5" in file:
        validation_files.append(LOAD_PATH + file)
    elif "dataset10" in file:
        validation_files.append(LOAD_PATH + file)
    elif "dataset15" in file:
        validation_files.append(LOAD_PATH + file)
    elif "dataset20" in file:
        validation_files.append(LOAD_PATH + file)
    elif "dataset25" in file:
        validation_files.append(LOAD_PATH + file)
    else:
        training_files.append(LOAD_PATH + file)

training_files = sorted(training_files)
validation_files = sorted(validation_files)
print("Num training files: ", len(training_files))
print("Num validation files: ", len(validation_files))

# extract samples from files
all_train_samples, all_train_labels, all_val_samples, all_val_labels = extract_samples(training_files,
                                                                                       validation_files,
                                                                                       KEYS,
                                                                                       DATA_TYPES,
                                                                                       False)
if CONV_NET and FEATURE_MODEL is not None:
    for d, train_samples in enumerate(all_train_samples):
        all_train_samples[d] = train_samples.reshape([-1, train_samples.shape[-1] // 3, 3])

    for d, val_samples in enumerate(all_val_samples):
        all_val_samples[d] = val_samples.reshape([-1, val_samples.shape[-1] // 3, 3])

# get outputs from model if required
if FEATURE_MODEL is not None:
    model = load_model(FEATURE_MODEL)
    model.summary()
    last_layer_model = Model(inputs=model.input, outputs=model.get_layer(LAYER_NAME).output)

    for d, train_samples in enumerate(all_train_samples):
        all_train_samples[d] = last_layer_model.predict(train_samples)

    for d, val_samples in enumerate(all_val_samples):
        all_val_samples[d] = last_layer_model.predict(val_samples)
    K.clear_session()

if DBN_MODEL is not None:
    pickle_in = open(DBN_MODEL, "rb")
    dbn = pickle.load(pickle_in)
    pickle_in.close()

    # calculate layerwise activations
    for d, train_samples in enumerate(all_train_samples):
        all_train_samples[d] = layerwise_activations(dbn, train_samples, num_activations=1)[LAYER_INDEX]

    for d, val_samples in enumerate(all_val_samples):
        all_val_samples[d] = layerwise_activations(dbn, val_samples, num_activations=1)[LAYER_INDEX]


def batch_generator(batch_size, sequence_length):
    """
    Generator function for creating random batches of training-data.
    """

    # Infinite loop.
    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, sequence_length, num_x_signals)
        x_batch = np.zeros(shape=x_shape, dtype=np.float32)

        # Allocate a new array for the batch of output-signals.
        y_shape = (batch_size, sequence_length, num_labels)
        y_batch = np.zeros(shape=y_shape, dtype=np.float32)

        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            # Get a random dataset-index:
            dataset_idx = np.random.randint(len(all_train_samples))
            # Get a random start-index.
            # This points somewhere into the training-data.
            start_idx = np.random.randint(all_train_samples[dataset_idx].shape[0] - sequence_length)

            # Copy the sequences of data starting at this index.
            x_batch[i] = all_train_samples[dataset_idx][start_idx:start_idx + sequence_length]
            y_batch[i] = all_train_labels[dataset_idx][start_idx:start_idx + sequence_length]

        yield (x_batch, y_batch)


def val_batch_generator(batch_size, sequence_length):
    """
    Generator function for creating random batches of training-data.
    """

    # Infinite loop.
    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, sequence_length, num_x_signals)
        x_batch = np.zeros(shape=x_shape, dtype=np.float32)

        # Allocate a new array for the batch of output-signals.
        y_shape = (batch_size, sequence_length, num_labels)
        y_batch = np.zeros(shape=y_shape, dtype=np.float32)

        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            # Get a random dataset-index:
            dataset_idx = np.random.randint(len(all_val_samples))
            # Get a random start-index.
            # This points somewhere into the training-data.
            start_idx = np.random.randint(all_val_samples[dataset_idx].shape[0] - sequence_length)

            # Copy the sequences of data starting at this index.
            x_batch[i] = all_val_samples[dataset_idx][start_idx:start_idx + sequence_length]
            y_batch[i] = all_val_labels[dataset_idx][start_idx:start_idx + sequence_length]

        yield (x_batch, y_batch)


num_x_signals = all_train_samples[0].shape[1]
num_labels = all_train_labels[0].shape[1]

save_path = os.path.join(SAVE_PATH, SAVE_NAME)
if not os.path.exists(save_path):
    os.makedirs(save_path)

generator = batch_generator(batch_size=BATCH_SIZE, sequence_length=NUM_TIME_STEPS)
val_generator = val_batch_generator(batch_size=BATCH_SIZE, sequence_length=NUM_TIME_STEPS)

a = Input(shape=(NUM_TIME_STEPS, num_x_signals,))
if BIDIRECTIONAL:
    x = Bidirectional(LSTM(units=LSTM_SIZE, return_sequences=True, dropout=FORWARD_DROPOUT, recurrent_dropout=RECURRENT_DROPOUT))(a)
else:
    x = LSTM(units=LSTM_SIZE, return_sequences=True, dropout=FORWARD_DROPOUT, recurrent_dropout=RECURRENT_DROPOUT)(a)
o = TimeDistributed(Dense(num_labels, activation='softmax'))(x)
model = Model(inputs=a, outputs=o)
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.summary()

checkpointer = ModelCheckpoint(filepath=os.path.join(save_path, 'best_model'),
                               save_best_only=True)
history = model.fit_generator(generator=generator, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH,
                              validation_data=val_generator, validation_steps=VALIDATION_STEPS_PER_EPOCH, verbose=1,
                              callbacks=[checkpointer])

pickle_out = open(os.path.join(save_path, 'history'), 'wb')
pickle.dump(history.history, pickle_out)
pickle_out.close()
K.clear_session()
