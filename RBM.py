import numpy as np
import tensorflow as tf
import train_utils
import deltas
import make_datasets as make_ds

if __name__ == '__main__':
    print('Using Tensorflow version: ', tf.__version__)


class RBM:
    def __init__(self, num_vunits, num_hunits, layer_type='bb', layer_index=0):
        self.num_vunits = num_vunits
        self.num_hunits = num_hunits

        self.layer_type = layer_type
        self.layer_index = layer_index

        # scale factor for weight initialization
        self.weight_scale = 1 / np.sqrt(self.num_vunits + self.num_hunits)
        # random uniform initialization
        self.weights = self.weight_scale * np.random.uniform(low=-1, high=1,
                                                             size=(self.num_vunits, self.num_hunits))
        # convert to float 32
        self.weights = self.weights.astype(np.float32)

        # initialize biases
        self.vbiases = np.zeros([1, self.num_vunits]).astype(np.float32)
        self.hbiases = np.zeros([1, self.num_hunits]).astype(np.float32)

        if self.layer_type == 'gb' or self.layer_type == 'gr':
            # initialize logarithmic standard deviation (z = log(sigma**2))
            self.log_sigmas = np.random.normal(loc=1.0, scale=0.01, size=(1, num_vunits)).astype(np.float32)

    def set_weights(self, weights):
        # initialize weights of RBM to some matrix of shape [num_vunits,num_hunits]
        if weights.shape != (self.num_vunits, self.num_hunits):
            print('Weights need to be shape [num_vunits, num_hunits]!')
            print('Weights not initialized!')
        else:
            self.weights = weights
            print('Weights initialized.')

    def set_vbiases(self, vbiases):
        # initialize vbiases of RBM to some matrix of shape [1,num_vunits]
        if vbiases.shape != (1, self.num_vunits):
            print('Visual biases need to be shape [1, num_vunits]!')
            print('Visual biases not initialized!')
        else:
            self.vbiases = vbiases
            print('Visubal biases initialized.')

    def set_hbiases(self, hbiases):
        # initialize vbiases of RBM to some matrix of shape [1,num_vunits]
        if hbiases.shape != (1, self.num_hunits):
            print('Hidden biases need to be shape [1, num_hunits]!')
            print('Hidden biases not initialized!')
        else:
            self.hbiases = hbiases
            print('Hidden biases initialized.')

    def train_rbm(self, train_data, epochs=10, batch_size=32, summary_path=None, summary_frequency=10,
                  dbn=None, update_vbiases=True, start_learning_rate=0.01, learning_rate_decay=(10, 1.0),
                  cd_steps=1, sparsity_rate=0.0, sparsity_goal=0.1, keys=None, data_types=None):

        # set up summary writer
        if summary_path:
            writer = tf.summary.FileWriter(summary_path)

        # build Tensorflow graph:

        # make datasets
        if isinstance(train_data, np.ndarray):
            # for train data in np array form
            train_dataset = make_ds.simple_dataset(train_data, batch_size, shuffle_buffer=100000, cache=True)

        elif isinstance(train_data, list):
            # for train data in list of filenames form
            train_dataset = make_ds.dataset_from_TFRecords(train_data, batch_size, keys, data_types,
                                                           shuffle_buffer=100000,
                                                           parallel_reads=len(train_data), num_cores=8)
        else:
            raise TypeError('train_data needs to be either numpy array or list of file paths')

        # initializable iterator
        iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

        # get batches from datasets
        if isinstance(train_data, list):
            batch, label_batch = iterator.get_next()

        else:
            batch = iterator.get_next()

        # define initialization operation
        train_init_op = iterator.make_initializer(train_dataset)

        # if RBM is part of dbn: upward propagation of inputs
        v0_ = train_utils.upward_propagation(batch, dbn, self.layer_index)

        # set up variables for training:
        # global step defines epoch for learning rate decay
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(start_learning_rate,
                                                   global_step,
                                                   learning_rate_decay[0],
                                                   learning_rate_decay[1],
                                                   staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        train_vbiases = tf.Variable(self.vbiases, name='vbiases')
        tf.summary.histogram("vbiases", train_vbiases)

        train_hbiases = tf.Variable(self.hbiases, name='hbiases')
        tf.summary.histogram("hbiases", train_hbiases)

        train_weights = tf.Variable(self.weights, name="weights")
        tf.summary.histogram("weights", train_weights)

        if self.layer_type == 'gb' or self.layer_type == 'gr':
            train_log_sigmas = tf.Variable(self.log_sigmas)
            tf.summary.histogram("logsigmas", train_log_sigmas)

        # reconstruction error
        err = tf.Variable(0, dtype=tf.float32)
        tf.summary.scalar('rec_error', err)

        # activations of all hidden units
        q = tf.Variable(tf.zeros([1, self.num_hunits], dtype=tf.float32), name='activations')
        tf.summary.histogram('activations', q)

        # average activation
        avg_activation = tf.Variable(0, dtype=tf.float32)
        tf.summary.scalar('avg_activation', avg_activation)

        # merge all summaries
        merged_summary = tf.summary.merge_all()

        # start training part of graph:
        increase_global_step = tf.assign_add(global_step, 1)

        # contrastive divergence procedure:
        with tf.name_scope('CD'):
            if self.layer_type == 'gb' or self.layer_type == 'gr':
                h0_, h0, vn_, vn, hn_ = train_utils.CD_procedure(v0_, self.layer_type, cd_steps,
                                                                 train_vbiases, train_hbiases,
                                                                 train_weights, train_log_sigmas)
            else:
                h0_, h0, vn_, vn, hn_ = train_utils.CD_procedure(v0_, self.layer_type, cd_steps,
                                                                 train_vbiases, train_hbiases, train_weights)

        with tf.name_scope('calculate_error'):
            assign_err = tf.assign(err, deltas.error(v0_, vn_))

        with tf.name_scope('activation'):
            assign_q = tf.assign(q, tf.reduce_mean(h0_, axis=0, keep_dims=True))
            assign_avg_activation = tf.assign(avg_activation, tf.reduce_mean(q))

        with tf.name_scope('sparsity'):
            if sparsity_rate != 0:
                if self.layer_type == 'gb' or self.layer_type == 'gr':
                    h_sparse_term, w_sparse_term = deltas.sparse_terms(sparsity_goal, q, h0_, v0_, train_log_sigmas)
                else:
                    h_sparse_term, w_sparse_term = deltas.sparse_terms(sparsity_goal, q, h0_, v0_)
            else:
                h_sparse_term, w_sparse_term = 0.0, 0.0

        with tf.name_scope('calculate_deltas'):
            if self.layer_type == 'gb' or self.layer_type == 'gr':
                if update_vbiases is True:
                    delta_vbiases = deltas.delta_vbiases(v0_, vn_, train_log_sigmas)
                else:
                    delta_vbiases = tf.zeros_like(train_vbiases)
                delta_hbiases = deltas.delta_hbiases(h0_, hn_) - sparsity_rate * h_sparse_term
                delta_weights = deltas.delta_weights(v0_, h0_, vn_, hn_,
                                                     train_log_sigmas) - sparsity_rate * w_sparse_term
                delta_log_sigmas = deltas.delta_log_sigmas(v0_, h0_, vn_, hn_,
                                                           train_vbiases,
                                                           train_weights,
                                                           train_log_sigmas)
            else:
                if update_vbiases is True:
                    delta_vbiases = deltas.delta_vbiases(v0_, vn_)
                else:
                    delta_vbiases = tf.zeros_like(train_vbiases)
                delta_hbiases = deltas.delta_hbiases(h0_, hn_) - sparsity_rate * h_sparse_term
                delta_weights = deltas.delta_weights(v0_, h0_, vn_, hn_) - sparsity_rate * w_sparse_term

        with tf.name_scope('updates'):
            assign_vbiases = tf.assign_add(train_vbiases, learning_rate * delta_vbiases)
            assign_hbiases = tf.assign_add(train_hbiases, learning_rate * delta_hbiases)
            assign_weights = tf.assign_add(train_weights, learning_rate * delta_weights)
            if self.layer_type == 'gb' or self.layer_type == 'gr':
                assign_log_sigmas = tf.assign_add(train_log_sigmas, learning_rate * delta_log_sigmas)

            # assign operations (are executed when session is run)
            assign_ops = [assign_err, assign_q, assign_avg_activation,
                          assign_hbiases, assign_weights]
            if self.layer_type == 'gb' or self.layer_type == 'gr':
                assign_ops.append(assign_log_sigmas)
            if update_vbiases is True:
                assign_ops.append(assign_vbiases)

        # Tensorflow session to execute graph:
        with tf.Session() as sess:
            # initialize all training variables
            sess.run(tf.global_variables_initializer())
            # finalize graph to prevent adding to it unintentionally
            tf.Graph.finalize(tf.get_default_graph())
            # initialize a counter for the batches trained
            batch_count = 0
            # Train for #epochs
            print('Starting training...')
            for step in range(epochs):
                print('Epoch {}/{}'.format(step, epochs))
                sess.run(train_init_op)
                sess.run(increase_global_step)
                while True:
                    try:
                        sess.run(assign_ops)
                        batch_count += 1
                        # write summaries to summary_path:
                        if summary_path:
                            if batch_count % summary_frequency == 0:
                                s = sess.run(merged_summary)
                                writer.add_summary(s, batch_count)
                    except tf.errors.OutOfRangeError:
                        break
                # assign new parameters after each epoch:
                self.vbiases = sess.run(train_vbiases)
                self.hbiases = sess.run(train_hbiases)
                self.weights = sess.run(train_weights)
                if self.layer_type == 'gb' or self.layer_type == 'gr':
                    self.log_sigmas = sess.run(train_log_sigmas)
        if summary_path:
            writer.close()
        tf.reset_default_graph()
