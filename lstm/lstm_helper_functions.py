import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
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


def get_accuracies_and_plot_labels(all_outputs, all_true_labels, time_window_length=1, save_path=None):
    """
    calculate accuracies and plot the per data labels
    :param all_outputs: list of outputs for each dataset, list of np.arrays of shape [num_samples, num_classes]
    :param all_true_labels: list of true labels for each dataset, list of np.arrays of shape [num_samples, num_classes]
    :param time_window_length: length of time window in seconds, int
    :param save_path: path to save label figure, string
    :return: reduced output classes and reduced true classes, list of np.arrays
    """
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
        outputs = all_outputs[d]
        output_classes = np.argmax(outputs, axis=1)

        # get true classes for one dataset:
        true_labels = all_true_labels[d][:outputs.shape[0], :]
        true_labels = true_labels.reshape([-1, 5])
        true_classes = np.argmax(true_labels, axis=1)

        # get accuracy
        num_corrects = np.where(np.array(true_classes) == np.array(output_classes))[0].shape[0]
        accuracy_before_average = num_corrects / len(output_classes)
        print(f'Accuracy dataset {d} before 30s average: ', accuracy_before_average)

        total_num_corrects += num_corrects
        total_num_samples += output_classes.shape[0]

        # make reduced classes
        output_classes_reduced = []
        true_classes_reduced = []
        samples_per_30s = int(30/time_window_length)
        for j in range(output_classes.shape[0] // samples_per_30s):
            values, counts = np.unique(output_classes[j*samples_per_30s:(j+1)*samples_per_30s], return_counts=True)
            max_index = np.argmax(counts)
            max_value = values[max_index]
            output_classes_reduced.append(max_value)

            values, counts = np.unique(true_classes[j*samples_per_30s:(j+1)*samples_per_30s], return_counts=True)
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
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'dataset_labeling.pdf'), format='pdf')
    plt.show()
    print('Mean Accuracy before average: ', total_num_corrects / total_num_samples)
    print('Mean Accuracy after average: ', num_corrects_after_average / num_samples_after_average)

    return all_output_classes_reduced, all_true_classes_reduced
