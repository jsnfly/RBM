import tensorflow as tf
import numpy as np
import train.make_datasets as make_ds
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K


def make_batches(sample_array, batch_size):
    sample_batches = []
    num_batches = sample_array.shape[0] // batch_size
    for batch_counter in range(num_batches):
        batch_samples = [sample_array[sample_counter * num_batches + batch_counter, :, :] for sample_counter in
                         range(batch_size)]
        batch = np.stack(batch_samples)
        sample_batches.append(batch)
    return sample_batches


def extract_samples(training_files, validation_files, keys, data_types, reverse=False):
    # make lists of numpy array where each array contains all samples from one dataset
    batches = []
    label_batches = []

    all_train_samples = []
    all_train_labels = []

    all_val_samples = []
    all_val_labels = []

    file_name = tf.placeholder(tf.string, shape=[None])
    dataset = make_ds.dataset_from_TFRecords(file_name, 500, keys, data_types,
                                             shuffle_buffer=0, num_cores=8, parallel_reads=1)
    iterator = dataset.make_initializable_iterator()
    sample_batch, label_batch = iterator.get_next()

    with tf.Session() as sess:
        for training_file in training_files:
            sess.run(iterator.initializer, feed_dict={file_name: [training_file]})
            while True:
                try:
                    samples, labels = sess.run([sample_batch, label_batch])
                    batches.append(samples)
                    label_batches.append(labels)
                except tf.errors.OutOfRangeError:
                    break
            samples = np.concatenate(batches)
            labels = np.concatenate(label_batches)

            batches = []
            label_batches = []

            if reverse is True:
                all_train_samples.append(np.flip(samples, axis=0))
                all_train_labels.append(np.flip(labels, axis=0))

            else:
                all_train_samples.append(samples)
                all_train_labels.append(labels)

        for validation_file in validation_files:
            sess.run(iterator.initializer, feed_dict={file_name: [validation_file]})
            while True:
                try:
                    samples, labels = sess.run([sample_batch, label_batch])
                    batches.append(samples)
                    label_batches.append(labels)
                except tf.errors.OutOfRangeError:
                    break
            samples = np.concatenate(batches)
            labels = np.concatenate(label_batches)

            batches = []
            label_batches = []

            if reverse is True:
                all_val_samples.append(np.flip(samples, axis=0))
                all_val_labels.append(np.flip(labels, axis=0))

            else:
                all_val_samples.append(samples)
                all_val_labels.append(labels)
    tf.reset_default_graph()

    return all_train_samples, all_train_labels, all_val_samples, all_val_labels

