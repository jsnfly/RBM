import tensorflow as tf
import numpy as np
import train.make_datasets as make_ds


def make_batches(sample_array, batch_size):
    sample_batches = []
    num_batches = sample_array.shape[0] // batch_size
    for batch_counter in range(num_batches):
        batch_samples = [sample_array[sample_counter * num_batches + batch_counter, :, :] for sample_counter in
                         range(batch_size)]
        batch = np.stack(batch_samples)
        sample_batches.append(batch)
    return sample_batches


def extract_samples(training_files, validation_files, keys, data_types):
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

            all_val_samples.append(samples)
            all_val_labels.append(labels)
    tf.reset_default_graph()

    return all_train_samples, all_train_labels, all_val_samples, all_val_labels

# # get features from feed forward network
# model_path = '/home/jonas/PycharmProjects/convnet/test2Model.hdf5'
# model = load_model(model_path)
# model.summary()
# layer_name = 'global_average_pooling1d'
# last_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
#
# for d, train_samples in enumerate(all_train_samples):
#     all_train_samples[d] = last_layer_model.predict(train_samples)
#
# for d, val_samples in enumerate(all_val_samples):
#     all_val_samples[d] = last_layer_model.predict(val_samples)
# K.clear_session()