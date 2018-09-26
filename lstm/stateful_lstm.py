import sys
sys.path.append("..")

import os
import pickle
import time
from evaluate.helper_functions import layerwise_activations
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed
from tensorflow.keras.optimizers import Adam
from lstm.lstm_helper_functions import *

# set paramters
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

NUM_TIME_STEPS = 60
LSTM_SIZE = 64
REVERSE = True
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
FORWARD_DROPOUT = 0
RECURRENT_DROPOUT = 0
SAVE_PATH = "Stateful_LSTM"
SAVE_NAME = f"512-512-64-Feedforward_{NUM_TIME_STEPS}length_{LSTM_SIZE}size_{int(time.time())}"
if REVERSE is True:
    SAVE_NAME = SAVE_NAME + "_Reverse"

# set FEATURE_MODEL to None if no keras model is used
FEATURE_MODEL = '/home/jonas/Desktop/pre_train_raw_data/512_512_64/unbalanced_old_and_new/finetune_unbalanced/run2Model.hdf5'
# LAYER_NAME can be obtained from calling model.summary()
LAYER_NAME = "dense_2"

# set DBN_MODEL to None if no dbn is used
DBN_MODEL = None
# LAYER_INDEX is only relevant for DBN models
LAYER_INDEX = -1  # -1 for last layer

if FEATURE_MODEL is not None and DBN_MODEL is not None:
    raise AttributeError("Keras model and DBN model given, set one or both to None!")

# get training and validation files
LOAD_PATH = "/home/jonas/HDD/data/unwindowed/unwindowed_z-transformed/"
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
                                                                                       REVERSE)

print("after file reading:\n___________________")
for s, l in zip(all_train_samples, all_train_labels):
    print(s.shape, l.shape)

print("val files: ")
for s, l in zip(all_val_samples, all_val_labels):
    print(s.shape, l.shape)

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

# cut all datasets to have a (whole number * BATCH_SIZE * time_steps) samples
num_samples_batch = BATCH_SIZE * NUM_TIME_STEPS

all_train_samples = [train_samples[:train_samples.shape[0] // num_samples_batch * num_samples_batch, :] for
                     train_samples in all_train_samples]

all_train_labels = [train_labels[:train_labels.shape[0] // num_samples_batch * num_samples_batch, :] for
                    train_labels in all_train_labels]

all_val_samples = [val_samples[:val_samples.shape[0] // num_samples_batch * num_samples_batch, :] for
                   val_samples in all_val_samples]

all_val_labels = [val_labels[:val_labels.shape[0] // num_samples_batch * num_samples_batch, :] for
                  val_labels in all_val_labels]

print("after cutting of unusable samples:\n___________________")
for s, l in zip(all_train_samples, all_train_labels):
    print(s.shape, l.shape)

print("val files: ")
for s, l in zip(all_val_samples, all_val_labels):
    print(s.shape, l.shape)

num_x_signals = all_train_samples[0].shape[1]
num_labels = all_train_labels[0].shape[1]

# reshape to match shape (num_series, num_time_steps, num_x_signals)
all_train_samples = [
    np.reshape(train_samples, [train_samples.shape[0] // NUM_TIME_STEPS, NUM_TIME_STEPS, num_x_signals]) for
    train_samples in all_train_samples]

all_train_labels = [np.reshape(train_labels, [train_labels.shape[0] // NUM_TIME_STEPS, NUM_TIME_STEPS, num_labels]) for
                    train_labels in all_train_labels]

all_val_samples = [np.reshape(val_samples, [val_samples.shape[0] // NUM_TIME_STEPS, NUM_TIME_STEPS, num_x_signals]) for
                   val_samples in all_val_samples]

all_val_labels = [np.reshape(val_labels, [val_labels.shape[0] // NUM_TIME_STEPS, NUM_TIME_STEPS, num_labels]) for
                  val_labels in all_val_labels]

print("after reshaping samples:\n___________________")
for s, l in zip(all_train_samples, all_train_labels):
    print(s.shape, l.shape)

print("val files: ")
for s, l in zip(all_val_samples, all_val_labels):
    print(s.shape, l.shape)

for counter, train_samples in enumerate(all_train_samples):
    batches = make_batches(train_samples, BATCH_SIZE)
    train_samples = np.concatenate(batches)
    all_train_samples[counter] = train_samples

for counter, train_labels in enumerate(all_train_labels):
    batches = make_batches(train_labels, BATCH_SIZE)
    train_labels = np.concatenate(batches)
    all_train_labels[counter] = train_labels

for counter, val_samples in enumerate(all_val_samples):
    batches = make_batches(val_samples, BATCH_SIZE)
    val_samples = np.concatenate(batches)
    all_val_samples[counter] = val_samples

for counter, val_labels in enumerate(all_val_labels):
    batches = make_batches(val_labels, BATCH_SIZE)
    val_labels = np.concatenate(batches)
    all_val_labels[counter] = val_labels

print("after making batches :\n___________________")
for s, l in zip(all_train_samples, all_train_labels):
    print(s.shape, l.shape)

print("val files: ")
for s, l in zip(all_val_samples, all_val_labels):
    print(s.shape, l.shape)

train_accuracies = []
test_accuracies = []
current_best_val_loss = 100

model = Sequential()
model.add(LSTM(units=LSTM_SIZE, return_sequences=True, batch_input_shape=(BATCH_SIZE, NUM_TIME_STEPS, num_x_signals),
               dropout=FORWARD_DROPOUT, recurrent_dropout=RECURRENT_DROPOUT, stateful=True))
model.add(TimeDistributed(Dense(5, activation="softmax")))

optimizer = Adam(LEARNING_RATE)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
model.summary()

save_path = os.path.join(SAVE_PATH, SAVE_NAME)
if not os.path.exists(save_path):
    os.makedirs(save_path)


lengths_train_datasets = [d.shape[0] for d in all_train_samples]
lengths_val_datasets = [d.shape[0] for d in all_val_samples]
for epoch in range(EPOCHS):
    epoch_tr_losses = []
    epoch_tr_accs = []
    for d, length in enumerate(lengths_train_datasets):
        train_samples = all_train_samples[d]
        train_labels = all_train_labels[d]
        num_batches = length // BATCH_SIZE
        for i in range(num_batches):
            x_batch = train_samples[i*BATCH_SIZE: (i+1)*BATCH_SIZE]
            y_batch = train_labels[i*BATCH_SIZE: (i+1)*BATCH_SIZE]
            tr_loss, tr_acc = model.train_on_batch(x_batch, y_batch)
            epoch_tr_losses.append(tr_loss)
            epoch_tr_accs.append(tr_acc)
        model.reset_states()
    train_accuracies.append(np.mean(epoch_tr_accs))

    epoch_test_losses = []
    epoch_test_accs = []
    for d, length in enumerate(lengths_val_datasets):
        val_samples = all_val_samples[d]
        val_labels = all_val_labels[d]
        num_samples = length // BATCH_SIZE
        for i in range(num_samples):
            x_batch = val_samples[i*BATCH_SIZE: (i+1)*BATCH_SIZE]
            y_batch = val_labels[i*BATCH_SIZE: (i+1)*BATCH_SIZE]
            test_loss, test_acc = model.test_on_batch(x_batch, y_batch)
            epoch_test_losses.append(test_loss)
            epoch_test_accs.append(test_acc)
        model.reset_states()
    test_accuracies.append(np.mean(epoch_test_accs))

    if np.mean(epoch_test_losses) < current_best_val_loss:
        current_best_val_loss = np.mean(epoch_test_losses)
        model.save(os.path.join(save_path, "best_model"))
        print("Model saved! Val acc: ", np.mean(epoch_test_accs), ", New best val loss: ", current_best_val_loss)

    else:
        print("Model not saved, Val acc: ", np.mean(epoch_test_accs))

pickle_out = open(os.path.join(save_path, "train_accuracies"), "wb")
pickle.dump(train_accuracies, pickle_out)
pickle_out.close()

pickle_out = open(os.path.join(save_path, "test_accuracies"), "wb")
pickle.dump(test_accuracies, pickle_out)
pickle_out.close()

K.clear_session()
