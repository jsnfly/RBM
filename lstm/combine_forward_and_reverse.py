import sys

sys.path.append("..")

import os
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential, load_model, Model
from evaluate.helper_functions import layerwise_activations
from evaluate.plot import plot_confusion_matrix
from tensorflow.keras.layers import Dropout, Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from lstm_helper_functions import *
from sklearn.metrics import confusion_matrix

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# print(device_lib.list_local_devices()

FORWARD_PATH = '/home/jonas/PycharmProjects/RBM/lstm/Stateful_LSTM/512-512-64-Feedforward_60length_64size_1537799658/best_model'
REVERSE_PATH = '/home/jonas/PycharmProjects/RBM/lstm/Stateful_LSTM/512-512-64-Feedforward_60length_64size_1537800016_Reverse/best_model'

# set paramters
NUM_TIME_STEPS = 60
BATCH_SIZE = 64
LSTM_SIZE = 64
SOFTMAX_DROPOUT = 0.5

SAVE_PATH = "Combined_forward_and_reverse_stateful"
SAVE_NAME = f"Feedforward_{NUM_TIME_STEPS}length_{LSTM_SIZE}size_{int(time.time())}"

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
                                                                                       False)

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

# cut all datasets to have a (whole number * BATCH_SIZE * time_steps) samples:
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

# get forward outputs:

# reshape to match shape (num_series, num_time_steps, num_x_signals):
all_train_samples_forward = [
    np.reshape(train_samples, [train_samples.shape[0] // NUM_TIME_STEPS, NUM_TIME_STEPS, num_x_signals]) for
    train_samples in all_train_samples]

all_val_samples_forward = [
    np.reshape(val_samples, [val_samples.shape[0] // NUM_TIME_STEPS, NUM_TIME_STEPS, num_x_signals]) for
    val_samples in all_val_samples]

for counter, train_samples in enumerate(all_train_samples_forward):
    batches = make_batches(train_samples, BATCH_SIZE)
    train_samples = np.concatenate(batches)
    all_train_samples_forward[counter] = train_samples

for counter, val_samples in enumerate(all_val_samples_forward):
    batches = make_batches(val_samples, BATCH_SIZE)
    val_samples = np.concatenate(batches)
    all_val_samples_forward[counter] = val_samples

length_train_datasets = [d.shape[0] for d in all_train_samples_forward]
length_val_datasets = [d.shape[0] for d in all_val_samples_forward]

model = load_model(FORWARD_PATH)
model.summary()
layer_name = 'lstm'
last_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

all_train_outputs = []
for d, length in enumerate(length_train_datasets):
    train_samples = all_train_samples_forward[d]
    num_samples = length // BATCH_SIZE
    train_outputs = []
    for i in range(num_samples):
        x_batch = train_samples[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        outputs = last_layer_model.predict_on_batch(x_batch)
        train_outputs.append(outputs)
    train_outputs = np.concatenate(train_outputs, axis=1)
    train_outputs = train_outputs.reshape([-1, LSTM_SIZE])
    last_layer_model.reset_states()
    all_train_outputs.append(train_outputs)

all_val_outputs = []
for d, length in enumerate(length_val_datasets):
    val_samples = all_val_samples_forward[d]
    num_samples = length // BATCH_SIZE
    val_outputs = []
    for i in range(num_samples):
        x_batch = val_samples[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        outputs = last_layer_model.predict_on_batch(x_batch)
        val_outputs.append(outputs)
    val_outputs = np.concatenate(val_outputs, axis=1)
    val_outputs = val_outputs.reshape([-1, LSTM_SIZE])
    last_layer_model.reset_states()
    all_val_outputs.append(val_outputs)
K.clear_session()

forward_train_outputs = all_train_outputs
forward_val_outputs = all_val_outputs

# get reverse outputs:

# flip datasets:
all_train_samples_reverse = [np.flip(train_samples, axis=0) for train_samples in all_train_samples]
all_val_samples_reverse = [np.flip(val_samples, axis=0) for val_samples in all_val_samples]

# reshape to match shape (num_series, num_time_steps, num_x_signals):
all_train_samples_reverse = [
    np.reshape(train_samples, [train_samples.shape[0] // NUM_TIME_STEPS, NUM_TIME_STEPS, num_x_signals]) for
    train_samples in all_train_samples_reverse]

all_val_samples_reverse = [
    np.reshape(val_samples, [val_samples.shape[0] // NUM_TIME_STEPS, NUM_TIME_STEPS, num_x_signals]) for
    val_samples in all_val_samples_reverse]

for counter, train_samples in enumerate(all_train_samples_reverse):
    batches = make_batches(train_samples, BATCH_SIZE)
    train_samples = np.concatenate(batches)
    all_train_samples_reverse[counter] = train_samples

for counter, val_samples in enumerate(all_val_samples_reverse):
    batches = make_batches(val_samples, BATCH_SIZE)
    val_samples = np.concatenate(batches)
    all_val_samples_reverse[counter] = val_samples

length_train_datasets = [d.shape[0] for d in all_train_samples_reverse]
length_val_datasets = [d.shape[0] for d in all_val_samples_reverse]

model = load_model(REVERSE_PATH)
model.summary()
layer_name = 'lstm'
last_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

all_train_outputs = []
for d, length in enumerate(length_train_datasets):
    train_samples = all_train_samples_reverse[d]
    num_samples = length // BATCH_SIZE
    train_outputs = []
    for i in range(num_samples):
        x_batch = train_samples[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        outputs = last_layer_model.predict_on_batch(x_batch)
        train_outputs.append(outputs)
    train_outputs = np.concatenate(train_outputs, axis=1)
    train_outputs = train_outputs.reshape([-1, LSTM_SIZE])
    last_layer_model.reset_states()
    all_train_outputs.append(train_outputs)

all_val_outputs = []
for d, length in enumerate(length_val_datasets):
    val_samples = all_val_samples_reverse[d]
    num_samples = length // BATCH_SIZE
    val_outputs = []
    for i in range(num_samples):
        x_batch = val_samples[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        outputs = last_layer_model.predict_on_batch(x_batch)
        val_outputs.append(outputs)
    val_outputs = np.concatenate(val_outputs, axis=1)
    val_outputs = val_outputs.reshape([-1, LSTM_SIZE])
    last_layer_model.reset_states()
    all_val_outputs.append(val_outputs)
K.clear_session()

# reflip
reverse_train_outputs = [np.flip(train_samples, axis=0) for train_samples in all_train_outputs]
reverse_val_outputs = [np.flip(val_samples, axis=0) for val_samples in all_val_outputs]

# train combined softmax layer:

# combine outputs
all_train_outputs_combined = [np.concatenate([x, y], axis=1) for x, y in
                              zip(forward_train_outputs, reverse_train_outputs)]
for train_outputs in all_train_outputs_combined:
    print(train_outputs.shape)

all_val_outputs_combined = [np.concatenate([x, y], axis=1) for x, y in zip(forward_val_outputs, reverse_val_outputs)]
for val_outputs in all_val_outputs_combined:
    print(val_outputs.shape)

model = Sequential()
model.add(Dropout(SOFTMAX_DROPOUT))
model.add(Dense(num_labels, input_shape=(2 * LSTM_SIZE,)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.fit(x=np.concatenate(all_train_outputs_combined, axis=0),
          y=np.concatenate(all_train_labels, axis=0),
          epochs=10,
          batch_size=64,
          validation_data=(np.concatenate(all_val_outputs_combined, axis=0), np.concatenate(all_val_labels, axis=0)),
          verbose=1, callbacks=[EarlyStopping()])

outputs = {}
for d, val_samples in enumerate(all_val_outputs_combined):
    outputs['dataset{}'.format(d)] = model.predict(val_samples)
    print(outputs['dataset{}'.format(d)].shape)
K.clear_session()

total_num_corrects = 0
total_num_samples = 0

num_corrects_after_average = 0
num_samples_after_average = 0

all_true_classes_reduced = []
all_output_classes_reduced = []

fig, axes = plt.subplots(2, 3, figsize=(5.79, 4.79))

for d in range(5):
    ax = axes.flat[d]
    # get output classes for one dataset:
    output = outputs['dataset{}'.format(d)]
    output = output.reshape([-1, 5])
    output_classes = np.argmax(output, axis=1)

    # get true classes for one dataset:
    true_labels = all_val_labels[d][:output.shape[0], :]
    true_classes = np.argmax(true_labels, axis=1)

    # get accuracy
    num_corrects = np.where(np.array(true_classes) == np.array(output_classes))[0].shape[0]
    accuracy_before_average = num_corrects / len(output_classes)
    print('Accuracy before 30s average: ', accuracy_before_average)

    total_num_corrects += num_corrects
    total_num_samples += len(output_classes)

    # make reduced classes
    output_classes_reduced = []
    true_classes_reduced = []
    for j in range(output_classes.shape[0] // 30):
        values, counts = np.unique(output_classes[j * 30:(j + 1) * 30], return_counts=True)
        max_index = np.argmax(counts)
        max_value = values[max_index]
        output_classes_reduced.append(max_value)

        values, counts = np.unique(true_classes[j * 30:(j + 1) * 30], return_counts=True)
        max_index = np.argmax(counts)
        max_value = values[max_index]
        true_classes_reduced.append(max_value)

    all_true_classes_reduced.append(true_classes_reduced)
    all_output_classes_reduced.append(output_classes_reduced)

    reduced_correct_inds = np.where(np.array(true_classes_reduced) == np.array(output_classes_reduced))[0]
    num_corrects = reduced_correct_inds.shape[0]
    accuracy_after_average = num_corrects / len(output_classes_reduced)
    num_corrects_after_average += num_corrects
    num_samples_after_average += len(output_classes_reduced)

    x = np.arange(len(true_classes_reduced))
    ax.plot(true_classes_reduced, lw=2, c='b', alpha=0.7, label='True Labels')
    ax.plot(x, output_classes_reduced, lw=2, c='r', alpha=0.7, label='Predicted Labels')
    print('Accuracy after 30s average: ', accuracy_after_average)
    ax.set_title('Test Dataset {}'.format(d + 1))
    ax.set_xlabel('Acc: {:03.1f} %'.format(accuracy_after_average * 100))
axes[1, 2].remove()
axes[0, 1].set_yticks([])
axes[0, 1].set_yticklabels([])
axes[0, 2].set_yticks([])
axes[0, 2].set_yticklabels([])
axes[1, 1].set_yticks([])
axes[1, 1].set_yticklabels([])
plt.tight_layout()
axes[1, 1].legend(loc=(1.2, 0.4))
# plt.savefig('/home/jonas/Schreibtisch/feedforward_and_fourier_labeling.pdf', format='pdf')
plt.show()
print('Mean Accuracy before average: ', total_num_corrects / total_num_samples)
print('Mean Accuracy after average: ', num_corrects_after_average / num_samples_after_average)

cm = confusion_matrix(np.concatenate(all_true_classes_reduced), np.concatenate(all_output_classes_reduced))
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
class_names = ['awake', 'N1', 'N2', 'N3', 'REM']
plt.figure(figsize=(5.79, 5.79))
plot_confusion_matrix(cm, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
plt.show()
