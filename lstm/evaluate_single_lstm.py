import sys

sys.path.append("..")

import os
import pickle
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model, Model
from evaluate.helper_functions import layerwise_activations
from evaluate.plot import plot_confusion_matrix
from tensorflow.keras import backend as K
from lstm.lstm_helper_functions import *
from sklearn.metrics import confusion_matrix

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

LSTM_PATH = '/home/jonas/Desktop/testing/stateful_reverse/512-256-128-64_finetuned_60length_64size_1538040177_Reverse/best_model'
SAVE_PATH = None
REVERSE = True
STATEFUL = True
NUM_TIME_STEPS = 60
WINDOW_LENGTH = 1  # length of single input samples, not whole time series,
# so e.g. samples of 256 values for a sampling rate of 256 Hz gives a window lenght of 1

BATCH_SIZE = 64
WARMUP_STEPS = 0

# set FEATURE_MODEL to None if no keras model is used
FEATURE_MODEL = '/home/jonas/Desktop/pre_train_raw_data/512_256_128_64/unbalanced_old_and_new/finetune_unbalanced/run1Model.hdf5'
# LAYER_NAME can be obtained from calling model.summary()
LAYER_NAME = "dense_3"

# set DBN_MODEL to None if no dbn is used
DBN_MODEL = None
# LAYER_INDEX is only relevant for DBN models
LAYER_INDEX = -1  # -1 for last layer

if FEATURE_MODEL is not None and DBN_MODEL is not None:
    raise AttributeError("Keras model and DBN model given, set one or both to None!")

LOAD_PATH = "/home/jonas/HDD/data/unwindowed/unwindowed_z-transformed/"
KEYS = ["sample", "one_hot_label"]
DATA_TYPES = ["float32", "int32"]

# get validation files
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

    if WARMUP_STEPS != 0:

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
            y_true_slice = y_true[:, WARMUP_STEPS:, :]
            y_pred_slice = y_pred[:, WARMUP_STEPS:, :]

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

    all_outputs = []
    for d, val_samples in enumerate(all_val_samples):
        outputs = []
        for s in range(val_samples.shape[0] // NUM_TIME_STEPS):
            sample = val_samples[s * NUM_TIME_STEPS:(s + 1) * NUM_TIME_STEPS, :]
            sample = np.expand_dims(sample, axis=0)
            output = model.predict(sample)
            outputs.append(output)
        outputs = np.concatenate(outputs, axis=1)
        outputs = outputs.reshape([-1, 5])
        all_outputs.append(outputs)
        print('shape outputs dataset{}'.format(d), all_outputs[d].shape)
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
    all_outputs = []
    for d, length in enumerate(lengths_val_datasets):
        outputs = []
        val_samples = all_val_samples[d]
        num_samples = length // BATCH_SIZE
        for i in range(num_samples):
            x_batch = val_samples[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            output = model.predict_on_batch(x_batch)
            outputs.append(output)
        model.reset_states()
        outputs = np.concatenate(outputs, axis=1)
        outputs = outputs.reshape([-1, 5])
        all_outputs.append(outputs)
        print('shape outputs dataset{}'.format(d), all_outputs[d].shape)
    K.clear_session()

    if REVERSE:
        # Reflip
        all_outputs = [np.flip(outputs, axis=0) for outputs in all_outputs]
        all_val_labels = [np.flip(val_labels, axis=0) for val_labels in all_val_labels]

all_output_classes_reduced, all_true_classes_reduced = get_accuracies_and_plot_labels(all_outputs,
                                                                                      all_val_labels,
                                                                                      time_window_length=WINDOW_LENGTH,
                                                                                      save_path=SAVE_PATH)

cm = confusion_matrix(np.concatenate(all_true_classes_reduced), np.concatenate(all_output_classes_reduced))
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
class_names = ['awake', 'N1', 'N2', 'N3', 'REM']
plt.figure(figsize=(5.79, 5.79))
plot_confusion_matrix(cm, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
plt.show()
