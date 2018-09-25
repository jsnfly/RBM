import sys

sys.path.append("..")

import os
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
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

LSTM_PATH = '/home/jonas/PycharmProjects/RBM/lstm/Stateful_LSTM/512-512-64-Feedforward_60length_64size_1537799658/best_model'
REVERSE = False
STATEFUL = True
NUM_TIME_STEPS = 60
BATCH_SIZE = 64

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
        pass

validation_files = sorted(validation_files)
print("Num validation files: ", len(validation_files))

# extract samples from files
all_train_samples, all_train_labels, all_val_samples, all_val_labels = extract_samples([],
                                                                                       validation_files,
                                                                                       KEYS,
                                                                                       DATA_TYPES,
                                                                                       REVERSE)
# get outputs from model if required
if FEATURE_MODEL is not None:
    model = load_model(FEATURE_MODEL)
    model.summary()
    last_layer_model = Model(inputs=model.input, outputs=model.get_layer(LAYER_NAME).output)
    for d, val_samples in enumerate(all_val_samples):
        all_val_samples[d] = last_layer_model.predict(val_samples)
    K.clear_session()

if DBN_MODEL is not None:
    pickle_in = open(DBN_MODEL, "rb")
    dbn = pickle.load(pickle_in)
    pickle_in.close()

    # calculate layerwise activations
    for d, val_samples in enumerate(all_val_samples):
        all_val_samples[d] = layerwise_activations(dbn, val_samples, num_activations=1)[LAYER_INDEX]

if not STATEFUL:
    # FOR STATELESS:
    WARMUP = False

    if WARMUP:
        warmup_steps = 10

        def loss_warmup(y_true, y_pred):
            """
            Calculate the Mean Squared Error between y_true and y_pred,
            but ignore the beginning "warmup" part of the sequences.

            y_true is the desired output.
            y_pred is the model's output.
            """

            # The shape of both input tensors are:
            # [batch_size, sequence_length, num_y_signals].

            # Ignore the "warmup" parts of the sequences
            # by taking slices of the tensors.
            y_true_slice = y_true[:, warmup_steps:, :]
            y_pred_slice = y_pred[:, warmup_steps:, :]

            # These sliced tensors both have this shape:
            # [batch_size, sequence_length - warmup_steps, num_y_signals]

            # Calculate the MSE loss for each value in these tensors.
            # This outputs a 3-rank tensor of the same shape.
            loss = tf.losses.softmax_cross_entropy(onehot_labels=y_true_slice,
                                                   logits=y_pred_slice)

            # Keras may reduce this across the first axis (the batch)
            # but the semantics are unclear, so to be sure we use
            # the loss across the entire tensor, we reduce it to a
            # single scalar with the mean function.
            loss_mean = tf.reduce_mean(loss)

            return loss_mean


        model = load_model(LSTM_PATH, custom_objects={'loss_warmup': loss_warmup})

    else:
        model = load_model(LSTM_PATH)

    outputs = {}
    for d, val_samples in enumerate(all_val_samples):
        outputs['dataset{}'.format(d)] = []
        for s in range(val_samples.shape[0] // NUM_TIME_STEPS):
            sample = val_samples[s * NUM_TIME_STEPS:(s + 1) * NUM_TIME_STEPS, :]
            sample = np.expand_dims(sample, axis=0)
            output = model.predict(sample)
            outputs['dataset{}'.format(d)].append(output)
        outputs['dataset{}'.format(d)] = np.concatenate(outputs['dataset{}'.format(d)], axis=1)
        print('shape outputs dataset{}'.format(d), outputs['dataset{}'.format(d)].shape)
    K.clear_session()

else:
    # cut all datasets to have a (whole number * BATCH_SIZE * time_steps) samples
    num_samples_batch = BATCH_SIZE * NUM_TIME_STEPS

    all_val_samples = [val_samples[:val_samples.shape[0] // num_samples_batch * num_samples_batch, :] for
                       val_samples in all_val_samples]

    all_val_labels = [val_labels[:val_labels.shape[0] // num_samples_batch * num_samples_batch, :] for
                      val_labels in all_val_labels]

    num_x_signals = all_val_samples[0].shape[1]
    num_labels = all_val_labels[0].shape[1]

    # reshape to match shape (num_series, num_time_steps, num_x_signals)
    all_val_samples = [np.reshape(val_samples, [val_samples.shape[0] // NUM_TIME_STEPS, NUM_TIME_STEPS, num_x_signals])
                       for
                       val_samples in all_val_samples]

    all_val_labels = [np.reshape(val_labels, [val_labels.shape[0] // NUM_TIME_STEPS, NUM_TIME_STEPS, num_labels]) for
                      val_labels in all_val_labels]

    for counter, val_samples in enumerate(all_val_samples):
        batches = make_batches(val_samples, BATCH_SIZE)
        val_samples = np.concatenate(batches)
        all_val_samples[counter] = val_samples

    print("val files: ")
    for s, l in zip(all_val_samples, all_val_labels):
        print(s.shape, l.shape)

    lengths_val_datasets = [d.shape[0] for d in all_val_samples]
    model = load_model(LSTM_PATH)
    outputs = {}
    for d, length in enumerate(lengths_val_datasets):
        outputs['dataset{}'.format(d)] = []
        val_samples = all_val_samples[d]
        num_samples = length // BATCH_SIZE
        for i in range(num_samples):
            x_batch = val_samples[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            output = model.predict_on_batch(x_batch)
            outputs['dataset{}'.format(d)].append(output)
        model.reset_states()
        outputs['dataset{}'.format(d)] = np.concatenate(outputs['dataset{}'.format(d)], axis=1)
        print('shape outputs dataset{}'.format(d), outputs['dataset{}'.format(d)].shape)
    K.clear_session()

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
    true_labels = true_labels.reshape([-1, 5])
    true_classes = np.argmax(true_labels, axis=1)

    if REVERSE:
        output_classes = np.flip(output_classes, axis=0)
        true_classes = np.flip(true_classes, axis=0)
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
