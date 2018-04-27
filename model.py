import json
import numpy as np
import os
import tensorflow as tf

from tqdm import tqdm


class CharacterModel():
    def __init__(self, data_provider, sess, config_file_path):
        """
        Config parameters:
            timesteps
            batch_size
            num_layers
            hidden_layer_size
            output_keep_prob
            input_keep_prob
            model_save_path
            model_summary_path
        """
        self.data_provider = data_provider
        self.sess = sess
        with open(config_file_path) as config_file:
            self.config = json.load(config_file)

        print("CONFIG:")
        print(self.config)

        self.build_model(self.config['training'])

    def assign_learning_rate(self, epoch):
        lr = float(self.config['learning_rate'])
        decay_rate = self.config['learning_rate_decay']
        after_epochs = self.config['learning_rate_decay_after']
        self.sess.run(tf.assign(self.learning_rate,
                                lr * (decay_rate ** max(0, epoch - after_epochs))))

    def build_model(self, training):
        if not training:
            self.config['timesteps'] = 1
            self.config['batch_size'] = 1

        self.config['vocab_size'] = self.data_provider.vocab_size

        self.inputs = tf.placeholder(dtype=tf.int32,
                                     shape=[None, self.config['timesteps']])
        inputs = tf.one_hot(self.inputs, self.config['vocab_size'])  # shape is now [batch_size, timesteps, vocab_size]
        inputs = tf.cast(inputs, tf.float32)

        self.initial_state = tf.placeholder(dtype=tf.float32,
                                            shape=[self.config['num_layers'], 2, None, self.config['hidden_layer_size']])
        self.sequence_lengths = tf.placeholder(dtype=tf.int32,
                                               shape=[None])

        if training:
            self.learning_rate = tf.Variable(self.config['learning_rate'], trainable=False)
            self.targets = tf.placeholder(dtype=tf.int32,
                                          shape=[None, self.config['timesteps']])
            self.weights = tf.placeholder(dtype=tf.float32,
                                          shape=[None, self.config['timesteps']])

        if training and self.config['output_keep_prob'] < 1.0:
            inputs = tf.nn.dropout(inputs,
                                   keep_prob=self.config['output_keep_prob'])

        states_per_layer = tf.unstack(self.initial_state,
                                      axis=0)
        initial_state = tuple(
            [tf.contrib.rnn.LSTMStateTuple(states_per_layer[layer][0],
                                           states_per_layer[layer][1])
             for layer in range(self.config['num_layers'])]
        )

        cells = []
        for _ in range(self.config['num_layers']):
            cell = tf.contrib.rnn.LSTMCell(self.config['hidden_layer_size'],
                                           forget_bias=1.0)
            if training and (self.config['input_keep_prob'] < 1.0 or self.config['output_keep_prob'] < 1.0):
                cell = tf.contrib.rnn.DropoutWrapper(cell,
                                                     input_keep_prob=self.config['input_keep_prob'],
                                                     output_keep_prob=self.config['output_keep_prob'])
            cells.append(cell)

        self.multi_cell = tf.contrib.rnn.MultiRNNCell(cells,
                                                      state_is_tuple=True)

        outputs, self.final_state = tf.nn.dynamic_rnn(cell=self.multi_cell,
                                                      inputs=inputs,
                                                      initial_state=initial_state,
                                                      dtype=tf.float32,
                                                      sequence_length=self.sequence_lengths)  # outputs is shape [batch_size, timesteps, num_hidden]
        # self.final_state is an LSTMStateTuple for each layer

        outputs = tf.reshape(outputs,
                             shape=[-1, self.config['hidden_layer_size']])  # shape [batch_size * timesteps, num_hidden]
        logits = tf.layers.dense(inputs=outputs,
                                 units=self.config['vocab_size'],
                                 activation=None,
                                 use_bias=True,
                                 kernel_initializer=tf.variance_scaling_initializer())
        self.logits = tf.reshape(logits,
                                 shape=[-1, self.config['timesteps'], self.config['vocab_size']])
        self.probs = tf.reshape(tf.nn.softmax(logits),
                                shape=[-1, self.config['timesteps'], self.config['vocab_size']])  # shape [batch_size, timesteps, vocab_size]

        # find the index of the maximum probability along the vocab_size axis (2) and cast to integer
        self.predictions = tf.cast(tf.argmax(self.logits, axis=2), tf.int32)
        self.predictions = tf.reshape(self.predictions,
                                      shape=[self.config['batch_size'], self.config['timesteps']])

        if training:
            self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.logits,
                                                         targets=self.targets,
                                                         weights=self.weights)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predictions, self.targets), tf.float32))
            self.build_optimizers()

    def build_summaries(self):
        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('loss', self.loss)
        summaries = tf.summary.merge_all()
        return summaries

    def build_optimizers(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        trainable_vars = tf.trainable_variables()
        gradients = tf.gradients(self.loss, trainable_vars)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.config['grad_clip'])
        self.optimizer_op = optimizer.apply_gradients(zip(clipped_gradients, trainable_vars))

    def clear_path(self, folder):
        if not os.path.exists(folder):
            return

        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                # elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)

    def get_zero_state(self, batch_size):
        return self.sess.run(self.multi_cell.zero_state(batch_size, dtype=tf.float32))

    def make_path(self, path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    def sample_model(self, num_chars_generate=200, primer='The '):
        initial_state = self.get_zero_state(1)
        sequence_lengths = np.ones([1]).astype(np.int32)
        for char in primer[:-1]:
            x = np.zeros([1, self.config['timesteps']])
            x[0, 0] = self.data_provider.convert_char(char)
            feed_dict = {
                self.inputs: x,
                self.initial_state: initial_state,
                self.sequence_lengths: sequence_lengths
            }
            initial_state = self.sess.run(self.final_state,
                                          feed_dict=feed_dict)

        sampled_string = primer
        char = primer[-1]
        for _ in range(num_chars_generate):
            x = np.zeros([1, self.config['timesteps']])
            x[0, 0] = self.data_provider.convert_char(char)
            feed_dict = {
                self.inputs: x,
                self.initial_state: initial_state,
                self.sequence_lengths: sequence_lengths
            }
            initial_state, probs = self.sess.run([self.final_state, self.probs],
                                                 feed_dict=feed_dict)
            char_index = np.random.choice(self.data_provider.vocab_size, p=probs[:, 0, :].flatten())
            char = self.data_provider.convert_char_index(char_index)
            sampled_string += char

        return sampled_string

    def save_model(self, iteration, max_to_keep=5):
        saver = tf.train.Saver(max_to_keep=max_to_keep)
        model_path = saver.save(self.sess, os.path.join(self.config['model_save_path'], "checkpoint.ckpt"),
                                global_step=iteration)
        print("Model saved in %s" % model_path)

    def train(self):
        sample_directory = os.path.dirname(self.config['sample_file'])
        self.clear_path(sample_directory)
        self.make_path(sample_directory)
        with open(self.config['sample_file'], 'w+'):
            pass  # make the sampling file so we can append to it

        self.clear_path(self.config['model_save_path'])
        self.make_path(self.config['model_save_path'])

        self.clear_path(self.config['model_summary_path'])
        self.make_path(self.config['model_summary_path'])

        merged_summaries = self.build_summaries()
        self.writer = tf.summary.FileWriter(self.config['model_summary_path'], self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        initial_state = self.get_zero_state(self.config['batch_size'])
        weights = np.ones([self.config['batch_size'], self.config['timesteps']])
        sequence_lengths = np.ones([self.config['batch_size']]).astype(np.int32) * int(self.config['timesteps'])

        epoch = 0
        iteration = 0
        while epoch < self.config['epochs']:
            x, y, reset = self.data_provider.sample_batch()
            if reset:
                epoch += 1
                self.assign_learning_rate(epoch)
                self.save_model(epoch)
                initial_state = self.get_zero_state(self.config['batch_size'])

            feed_dict = {
                self.inputs: x,
                self.targets: y,
                self.initial_state: initial_state,
                self.weights: weights,
                self.sequence_lengths: sequence_lengths
            }

            initial_state, loss, accuracy, summary, _ = self.sess.run([self.final_state, self.loss, self.accuracy, merged_summaries,
                                                                       self.optimizer_op],
                                                                      feed_dict=feed_dict)
            if iteration % self.config['summary_iterations'] == 0:
                self.writer.add_summary(summary, iteration)

            if iteration % self.config['sample_iterations'] == 0:
                sampled_string = self.sample_model()
                self.write_sampled_string(sampled_string, iteration)

            self.writer.add_summary(summary, iteration)
            iteration += 1

    def write_sampled_string(self, sampled_string, iteration):
        with open(self.config['sample_file'], 'a') as sample_file:
            sample_file.write(str(iteration) + ' BEGIN:' + '-' * 70 + '\n')
            sample_file.write(sampled_string + '\n')
            sample_file.write(str(iteration) + ' END  :' + '-' * 70 + '\n')
