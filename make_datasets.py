import tensorflow as tf


def simple_dataset(training_samples, batch_size, shuffle_buffer=0, cache=True):
    '''
    prepare a dataset from numpy array without applying a function to it
    Args:
        training_samples: numpy array of training samples, shape: [num_samples,num_vUnits]
        batch_size: size of a mini-batch, int
        shuffle_buffer: number of examples that are shuffled simultaniously, 0 means no shuffeling
        cache: whether to cache the dataset, should be set to False only for large datasets
    Returns:
        dataset: (shuffled) and batched dataset
    '''
    dataset = tf.data.Dataset.from_tensor_slices(training_samples)
    if shuffle_buffer != 0:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=1)
    if cache == True:
        dataset = dataset.cache()
    return dataset


def flatten1(x):
    '''
    flatten tensor x
    -> results in [sample_ch1,sample_ch2,sample_ch3,sample_ch1,sample_ch2,...]
    '''
    flat = tf.reshape(x, [-1])
    return flat


def flatten2(x):
    '''
    flatten tensor x, column major
    -> results in [samples_ch1,samples_ch2,samples_ch3]
    '''
    flat = tf.reshape(tf.transpose(x), [-1])
    return flat


def sliding_window_dataset(training_samples, window_size, batch_size, stride=1, num_cores=4):
    '''
    prerpare a dataset by slinding an input window over the samples
    Args:
        training_samples: numpy array of training samples, shape: [num_samples,num_channels]
        batch_size: size of a mini-batch, int
        num_cores: number of CPU cores available in the system
    Returns:
        dataset: (shuffled) and batched sliding window dataset
    '''
    dataset = tf.data.Dataset.from_tensor_slices(training_samples)
    dataset = dataset.apply(tf.contrib.data.sliding_window_batch(window_size, stride=stride))
    dataset = dataset.map(map_func=flatten2, num_parallel_calls=num_cores)
    dataset = dataset.batch(batch_size)
    return dataset


def make_one_hot_window_label(sliding_window_sample):
    '''
    helper function for sliding_window_dataset_labels,
    casts labels to one hot lables and then transforms a window with window_size labels
    into just a single label for the whole window
    '''
    one_hot_labels = tf.one_hot(sliding_window_sample, depth=5, dtype=tf.int32)
    reshaped = tf.reshape(one_hot_labels, [-1, one_hot_labels.shape[-1]])
    #   mean = tf.reduce_mean(reshaped,axis=0)
    mean = tf.reduce_sum(reshaped, axis=0) / tf.shape(reshaped)[0]
    _, ind = tf.nn.top_k(mean, k=1)
    one_hot_label = tf.scatter_nd([ind], [1], shape=mean.shape)
    return one_hot_label


def sliding_window_dataset_labels(one_hot_labels, window_size, batch_size, stride=1, num_cores=4):
    '''
    prerpare a dataset by sliding an input window over the labels
    Args:
        one_hot_labels: numpy array of training labels, shape: [num_samples,num_classes]
        batch_size: size of a mini-batch, int
        num_cores: number of CPU cores available in the system
    Returns:
        dataset: (shuffled) and batched sliding window dataset
    '''
    dataset = tf.data.Dataset.from_tensor_slices(one_hot_labels)
    dataset = dataset.apply(tf.contrib.data.sliding_window_batch(window_size, stride=stride))
    dataset = dataset.map(map_func=make_one_hot_window_label, num_parallel_calls=num_cores)
    dataset = dataset.batch(batch_size)
    return dataset


def _bytes_feature(value):
    '''
    helper funtion to convert value into a bytes feature
    '''
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_to_TFRecord(file_name, keys_and_raw_features):
    '''
    write features to TFRecords file
    Args:
        file_name: name of TFRecords file
        keys_and_raw_features: dictionary, where keys are the names of the features
        and the corresponding values are the features
    '''
    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(file_name)
    for i in range(len(keys_and_raw_features[list(keys_and_raw_features.keys())[0]])):
        feature = {}
        for key in keys_and_raw_features.keys():
            value = keys_and_raw_features[key][i].tobytes()
            feature[key] = _bytes_feature(value)
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # Serialize to string and write to the file
        writer.write(example.SerializeToString())
    writer.close()


def dataset_from_TFRecords(file_name, batch_size, keys, data_types, shuffle_buffer=0, parallel_reads=1, num_cores=4):
    '''
    returns features with names specified in keys with types from data_types
    Args:
        file_name: name of TFRecords file, string
        batch_size: size of one mini-batch, int
        keys: keys of the features to load, list of strings
        data_types: list of datatypes corresponding to keys, list of strings ('float32' or 'int32')
        (must be the datatypes which where used when writing the file!!)
        shuffle_buffer: number of examples that are shuffled simultaniously, 0 means no shuffeling
        parallel_reads: number of files that are read in parallel, int
        num_cores: number of CPU cores available in the system
    '''

    def parse(serialized):
        '''
        helper function: convert dataset from TFRecords file
        '''
        features = {}
        for key in keys:
            features[key] = tf.FixedLenFeature([], tf.string)

        # Parse the serialized data so we get a dict with our data.
        parsed_example = tf.parse_single_example(serialized=serialized,
                                                 features=features)
        results = []
        for i in range(len(keys)):
            key = keys[i]
            dtype = data_types[i]
            # Get the image as raw bytes.
            raw_feature = parsed_example[key]
            # Decode the raw bytes so it becomes a tensor with type.
            if dtype == 'float32':
                result_feature = tf.decode_raw(raw_feature, tf.float32)
            else:
                result_feature = tf.decode_raw(raw_feature, tf.int32)
            results.append(result_feature)
        return results

    dataset = tf.data.TFRecordDataset(file_name, num_parallel_reads=parallel_reads)
    dataset = dataset.map(map_func=parse, num_parallel_calls=num_cores)
    if shuffle_buffer != 0:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    dataset = dataset.batch(batch_size)
    return dataset


def make_LSH_values_and_indicies(batch, random_binary_matrix, num_KCs, p_WTA, return_WTA_matrix=False):
    '''
    return values and indicies of WTA_activation
    Args:
        batch: batch to calculate WTA activations from, shape [batch_size,num_vUnits]
        random_binary_matrix: matrix that connects input to Kenyon cells,
            tensorflow constant, shape [num_vUnits,num_KCs]
        num_KCs: number of Kenyon cells, int
        p_WTA: percentage of KCs that do not get silicend, float
        return_WTA_matrix: whether to calculate the WTA_matrix (for testing purposes), bool
    Returns:
        WTA_values: values of WTA units, shape [batch_size,num_activations]
        WTA_indices_flat: indices of WTA units, shape [batch_size,num_activations]
    '''
    num_activations = int(p_WTA * num_KCs)
    activation_KCs = tf.matmul(batch, random_binary_matrix)
    WTA_values, WTA_indices_flat = tf.nn.top_k(activation_KCs, k=num_activations)
    if return_WTA_matrix == True:
        batch_size = tf.shape(batch)[0]
        WTA_indices = tf.stack([tf.stack([tf.range(start=0, limit=batch_size) for i in range(num_activations)], axis=1),
                                WTA_indices_flat], axis=-1)
        WTA_indices = tf.reshape(WTA_indices, [-1, 2])
        WTA_values_flat = flatten1(WTA_values)
        activation_WTAs = tf.scatter_nd(WTA_indices, WTA_values_flat, shape=[batch_size, num_KCs])
        return activation_WTAs, WTA_values, WTA_indices_flat
    else:
        return WTA_values, WTA_indices_flat


def WTA_activations_from_values_and_indices(values, indices, num_KCs, p_WTA):
    '''
    get WTA matrix from values and indicies
    Args:
        values: values of WTA units, shape [batch_size,num_activations]
        indices: indices of WTA units, shape [batch_size,num_activations]
        num_KCs: number of Kenyon cells, int
        p_WTA: percentage of WTA cells, float
    Returns:
        WTA_rec: reconstructed matrix, shape [batch_size,num_KCs]
    '''
    num_activations = int(p_WTA * num_KCs)
    batch_size = tf.shape(values)[0]
    indices = tf.stack([tf.stack([tf.range(start=0, limit=batch_size) for i in range(num_activations)], axis=1),
                        indices], axis=-1)
    values = flatten1(values)
    indices = tf.reshape(indices, [-1, 2])
    WTA_rec = tf.scatter_nd(indices, values, shape=[batch_size, num_KCs])
    return WTA_rec


def WTA_activations_from_values_and_indices_map_fn(values, indices, labels):
    '''
    same as WTA_activations_from_values_and_indices but for a dataset
    '''
    num_KCs = 16 * 3 * 256
    p_WTA = 0.05

    num_activations = int(p_WTA * num_KCs)
    batch_size = tf.shape(values)[0]
    indices = tf.stack([tf.stack([tf.range(start=0, limit=batch_size) for i in range(num_activations)], axis=1),
                        indices], axis=-1)
    values = flatten1(values)
    indices = tf.reshape(indices, [-1, 2])
    WTA_rec = tf.scatter_nd(indices, values, shape=[batch_size, num_KCs])
    return WTA_rec, labels
