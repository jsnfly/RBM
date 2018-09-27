import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import train.make_datasets as make_ds
import os
import random
from prepare_data.data_helper_functions import *

print(tf.__version__)  # >1.7 is required

# set parameters
LABEL_SEC = 30  # number of seconds corresponding to one label
SAMPLE_FREQ = 256  # number of sample points per second
START_CUT_OFF = 0  # cut off first x hours
# START_CUT_OFF in seconds should be integer multiple
# of LABEL_SEC

DATA_PATH = '/home/jonas/HDD/data/raw_data/'
SAVE_PATH = '/home/jonas/Desktop/testing/fourier_data_nostride_nopreemph'

SHUFFLE = False
FOURIER_TRANSFORM = True
APPLY_SIGMOID = False
STRIDE = 0  # set to zero for no stride
PRE_EMPHASIS = 0

WINDOW_SIZE = 256
BATCH_SIZE = 512
SAMPLES_PER_FILE = 1000000

start = 7 + START_CUT_OFF * SAMPLE_FREQ * 3600  # start = header + START_CUT_OFF*SAMPLE_FREQ*3600seconds
start_labels = 0 + START_CUT_OFF * 3600 / LABEL_SEC  # start = header + START_CUT_OFF*3600seconds/LABEL_SEC

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# get files
load_paths = []
for filepath, directories, files in os.walk(DATA_PATH):
    for directory in directories:
        load_paths.append(os.path.join(DATA_PATH, directory))
load_paths = sorted(load_paths)
print("Num datasets: ", len(load_paths))
print("Directories: ", load_paths)

for load_path in load_paths:
    load_path = os.path.normpath(load_path)
    file_name = load_path.split(os.sep)[-1]

    print('File: ', file_name)
    print('____________________________________')
    batches = []
    label_batches = []

    C4M1 = txtfile_to_nparray(os.path.join(load_path, 'C4M1.txt'), start)
    F4M1 = txtfile_to_nparray(os.path.join(load_path, 'F4M1.txt'), start)
    O2M1 = txtfile_to_nparray(os.path.join(load_path, 'O2M1.txt'), start)

    data = np.stack([C4M1, F4M1, O2M1], axis=1).astype(np.float32)
    print('Data shape: ', np.shape(data))

    # read in labels
    labels = txtfile_to_nparray(os.path.join(load_path, 'labels.txt'), start_labels)

    print('Labels shape: ', np.shape(labels))
    num_label_samples = np.shape(labels)[0] * LABEL_SEC * SAMPLE_FREQ
    print('Resulting number of samples: ', num_label_samples)

    # number of labels of each class
    unique, counts = np.unique(labels, return_counts=True)
    print('Label distribution: ', dict(zip(unique, counts)))

    # indecies of artefact labels
    ind_artefact_label, = np.where(labels == -1.0)
    print('Artefact label indecies: ', ind_artefact_label)

    # delete artefact labels
    labels = np.delete(labels, ind_artefact_label)
    print('Labels shape after deleting artefacts: ', np.shape(labels))

    # start indecies of artefact samples
    ind_artefact_sample = ind_artefact_label * LABEL_SEC * SAMPLE_FREQ
    print('Artefact sample start indecies: ', ind_artefact_sample)

    # indecies of artefact samples
    inds_expanded = []
    for ind in ind_artefact_sample:
        ind_expanded = np.arange(ind, ind + LABEL_SEC * SAMPLE_FREQ)
        inds_expanded.extend(ind_expanded)
    print('Shape artefact sample indecies: ', np.shape(inds_expanded))

    data = np.delete(data, inds_expanded, axis=0)
    print('Data shape after artifact deletion: ', np.shape(data))

    # number of labels that have to be cut off
    num_labels_co = math.ceil(np.shape(labels)[0] - np.shape(data)[0] / LABEL_SEC / SAMPLE_FREQ)
    print('Number of labels that have to be cut off: ', num_labels_co)

    # cut off labels
    if num_labels_co > 0:
        labels = labels[:-num_labels_co]

    # cut samples to match number of samples expected from labels
    num_expected_samples = np.shape(labels)[0] * LABEL_SEC * SAMPLE_FREQ
    if np.shape(data)[0] > num_expected_samples:
        data = data[:num_expected_samples, :]

    # expand each label to size LABEL_SEC*SAMPLE_FREQ
    labels_expanded = []
    for l in labels:
        l_part = np.full(LABEL_SEC * SAMPLE_FREQ, fill_value=l)
        labels_expanded.append(l_part)

    labels_expanded = np.concatenate(labels_expanded, axis=0)
    print('Labels expanded shape: ', np.shape(labels_expanded), 'Dtype: ', labels_expanded.dtype)
    print('Final Data shape: ', np.shape(data), 'Dtype: ', data.dtype)

    if PRE_EMPHASIS != 0:
        # pre-emphasis (enhance high frequencies):
        # http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
        for i in range(np.shape(data)[1]):
            data[1:, i] = data[1:, i] - PRE_EMPHASIS * data[:-1, i]

    # Z-trafo
    for i in range(np.shape(data)[1]):
        mean = np.mean(data[:, i])
        std_dev = np.std(data[:, i])
        data[:, i] = (data[:, i] - mean) / std_dev

    # make sliding window samples and labels
    # SHUFFLE should be false, else labels no longer fit!!
    dataset = make_ds.sliding_window_dataset(data, WINDOW_SIZE, BATCH_SIZE, stride=STRIDE,
                                             num_cores=8)
    label_dataset = make_ds.sliding_window_dataset_labels(labels_expanded, WINDOW_SIZE, BATCH_SIZE, stride=STRIDE,
                                                          num_cores=8)

    iterator = dataset.make_one_shot_iterator()
    label_iterator = label_dataset.make_one_shot_iterator()
    batch = iterator.get_next()
    label_batch = label_iterator.get_next()

    if APPLY_SIGMOID:
        batch = tf.nn.sigmoid(batch)

    if FOURIER_TRANSFORM:
        batch, label_batch = make_fourier_trafo_sample(batch, label_batch)

    batch_counter = 0
    file_counter = 0
    with tf.Session() as sess:
        while True:
            try:
                batch_, label_batch_ = sess.run([batch, label_batch])
                batches.append(batch_)
                label_batches.append(label_batch_)
                batch_counter += 1
                if batch_counter * BATCH_SIZE >= SAMPLES_PER_FILE:
                    if SHUFFLE:
                        joined = list(zip(batches, label_batches))
                        random.shuffle(joined)
                        batches, label_batches = zip(*joined)
                    samples = np.concatenate(batches)
                    labels = np.concatenate(label_batches)
                    keys_and_raw_features = {'sample': samples,
                                             'one_hot_label': labels}
                    make_ds.write_to_TFRecord(os.path.join(SAVE_PATH, file_name + f'({file_counter}).tfrecords'),
                                              keys_and_raw_features)
                    batches = []
                    label_batches = []
                    batch_counter = 0
                    file_counter += 1
            except tf.errors.OutOfRangeError:
                if SHUFFLE:
                    joined = list(zip(batches, label_batches))
                    random.shuffle(joined)
                    batches, label_batches = zip(*joined)
                samples = np.concatenate(batches)
                labels = np.concatenate(label_batches)
                keys_and_raw_features = {'sample': samples,
                                         'one_hot_label': labels}
                make_ds.write_to_TFRecord(os.path.join(SAVE_PATH, file_name + f'({file_counter}).tfrecords'),
                                          keys_and_raw_features)
                break
    tf.reset_default_graph()
