"""

Variable notations:
s: synaptic connection weight
c: synaptic connection probability
b: leak bias

"""
import os
import shutil
import numpy as np
import tensorflow as tf
from itertools import permutations

from .truenorth import TrueNorthChip, NEURON_DEST_OUT
from .util import maybe_mkdir

# Default types for tensors
TF_DTYPE = tf.float32
NP_DTYPE = np.float32
NP_DTYPE_INT = np.int32

TF_MODEL_NAME = 'model.ckpt'


def normal_ccdf(x, mu, sigma2):
    """Normal CCDF"""
    # Check for degenerate distributions when sigma2 == 0
    # if x >= mu, n = 0
    # if x < mu, n = 1
    # sigma2_le_0 = tf.less_equal(sigma2, 0.)
    # x_gte_mu = tf.greater_equal(x, mu)
    # x_lt_mu = tf.less(x, mu)

    # Never divide by zero, instead the logic below handles degenerate distribution cases
    # sigma2 = tf.cond(sigma2_le_0, lambda: tf.ones_like(sigma2), lambda: sigma2)

    p = (1. - 0.5 * (1. + tf.erf((x - mu) / tf.sqrt(2. * sigma2))))
    # p = tf.cond(tf.logical_and(sigma2_le_0, x_gte_mu), lambda: tf.zeros_like(p), lambda: p)
    # p = tf.cond(tf.logical_and(sigma2_le_0, x_lt_mu), lambda: tf.ones_like(p), lambda: p)
    return p


def leak_bias(shape, core_name):
    """Create a bias variable, which is implemented as a leak"""
    return tf.Variable(tf.zeros(shape, dtype=TF_DTYPE), name='%s_leak_bias' % core_name)


def synapse_connection(shape, core_name):
    """Create a variable for the synaptic connection probability"""
    return tf.Variable(tf.random_uniform(shape, 0., 1., dtype=TF_DTYPE,
                                         seed=np.random.randint(0, np.iinfo(np.int32).max)),
                       name='%s_synapse_connection' % core_name)


def synapse_weight(shape, core_name, weights=np.array([-2, -1, 1, 2])):
    """Create a constant for the synaptic connection weights, which remain fixed during training"""
    # Evenly distribute the 4 axon types by first choosing the axon types
    axon_types = np.repeat(np.arange(4, dtype=NP_DTYPE_INT), shape[0] / 4)[:shape[0]]
    axon_types = np.random.permutation(axon_types)

    # Then compute all possible permutations from the weight choices
    axon_weight_perms = np.array(list(permutations(weights)), dtype=NP_DTYPE_INT)

    # And index into the array of permutations
    axon_weight_idx = np.tile(np.arange(len(axon_weight_perms)), np.ceil(shape[1] / len(axon_weight_perms)))[:shape[1]]
    axon_weight_idx = np.random.permutation(axon_weight_idx)

    axon_weights = axon_weight_perms[axon_weight_idx]

    # Create the full synaptic weight matrix from the types and weights
    s = axon_weights[:, axon_types].T.astype(NP_DTYPE)

    return tf.Variable(s, name='%s_synapse_weight' % core_name, trainable=False), \
           tf.Variable(axon_types, name='%s_axon_type' % core_name, trainable=False), \
           tf.Variable(axon_weights, name='%s_axon_weight' % core_name, trainable=False)


def connect_cores(input, output_dim, name):
    """Connect two cores given the inputs, synaptic weights, and output dimension.
    Inputs can be output from a previous core or spike inputs"""
    input_dim = int(input.get_shape()[1])

    s, axon_types, axon_weights = synapse_weight((input_dim, output_dim), name)
    b = leak_bias([output_dim], name)
    c = synapse_connection([input_dim, output_dim], name)

    xc = tf.reshape(input, (-1, input_dim, 1)) * c
    mu = b + tf.reduce_sum(xc * s, 1)
    sigma2 = tf.reduce_sum(xc * (1. - xc) * tf.pow(s, 2), 1)

    # Output is proba that each neuron fires
    x0 = tf.zeros_like(mu)
    output = normal_ccdf(x0, mu, sigma2)

    return output, b, c, axon_types, axon_weights, s


def configure_core(core, b, c, axon_types, axon_weights, dest_core, dest_axon):
    """Configure a TrueNorth core, given the corresponding parameters of the shadow network"""
    num_axons, num_neurons = c.shape

    # Set the axon types and weights, only up to the neurons for this core
    core.sigma[:num_neurons, :] = np.sign(axon_weights)
    core.s[:num_neurons, :] = np.abs(axon_weights)

    # Use the bias as a leak
    core.sigma_lambda[:num_neurons] = np.sign(b)
    core.lambda_[:num_neurons] = np.abs(b)
    core.sigma_lambda[core.sigma_lambda == 0] = 1

    # Most of these correspond to default values and shouldn't change
    core.b[:] = False
    core.c_lambda[:] = False
    core.epsilon[:] = False
    core.alpha[:] = 0
    core.beta[:] = 0
    core.TM[:] = 0
    core.gamma[:] = 0
    core.kappa[:] = True
    core.sigma_VR[:] = 1
    core.VR[:] = 0
    core.V[:] = 0

    # Set the axon types and weights
    core.axon_type[:num_axons] = axon_types

    # Set the synaptic connections
    core.w[:num_axons, :num_neurons] = c > np.random.uniform(0, 1, size=c.shape)

    core.dest_core[:num_neurons] = dest_core
    core.dest_axon[:num_neurons] = dest_axon

    return core


class TrueShadow(object):
    """The probabilistic interpretation of a spiking network"""

    def __init__(self):
        """The model must either be loaded or trained to be tested or deployed"""
        self.sess = None

    def load(self, dirpath, model_name=TF_MODEL_NAME):
        """Load the model from a checkpoint dir"""
        model_fname = os.path.join(dirpath, model_name)
        saver = tf.train.Saver()
        sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state(dirpath)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            self.sess = sess
        else:
            print('Unable to load from checkpoint:', model_fname)

        return

    def train(self, train_data, validation_data=None,
              batch_size=100, num_epochs=10,
              checkpoint_steps=1, report_steps=1, deploy_steps=0,
              dirpath=None, model_name=TF_MODEL_NAME, dropout=0.5):
        """Train the network and optionally save checkpoints"""

        if dirpath:
            ckpt_dir = maybe_mkdir(dirpath)
            model_fname = os.path.join(ckpt_dir, model_name)

        # Op to initialize all variables which should have been created in a subclass initialization
        init_op = tf.global_variables_initializer()

        # Create a saver
        saver = tf.train.Saver()

        # Create the session and save it for future use. Only other place sessions are created is in .load()
        sess = tf.Session()

        sess.run(init_op)
        self.sess = sess

        train_size = len(train_data)
        tf.train.start_queue_runners(sess=sess)

        results = []
        for epoch_i in range(1, num_epochs + 1):
            for batch_i in range(train_size // batch_size):
                batch_xs, batch_ys = train_data.next_batch(batch_size)

                sess.run(self.optimizer,
                         feed_dict={
                             self.x: batch_xs,
                             self.y: batch_ys,
                             self.keep_prob: dropout,
                         })

                # Clip the bias variables to [-255, 255] and connection probas to [0, 1]
                for (core_b, core_c, _, _, _) in self.core_params.values():
                    sess.run(tf.assign(core_b, tf.clip_by_value(core_b, -255., 255.)))
                    sess.run(tf.assign(core_c, tf.clip_by_value(core_c, 0., 1.)))

            report = ''
            report_values = [epoch_i]

            if report_steps > 0 and (epoch_i % report_steps == 0 or epoch_i == num_epochs):
                train_acc = self.test(train_data, batch_size=batch_size)
                test_acc = self.test(validation_data, batch_size=batch_size)
                report += '[ Train  ACC: %.4f ][ Test  ACC: %.4f ]'
                report_values.append(train_acc)
                report_values.append(test_acc)

            if deploy_steps > 0 and (epoch_i % deploy_steps == 0 or epoch_i == num_epochs):
                train_spike_acc = self.deploy(validation_data)
                test_spike_acc = self.deploy(validation_data)
                report += '[ Train Spike ACC: %.4f ][ Test Spike ACC: %.4f ]'
                report_values.append(train_spike_acc)
                report_values.append(test_spike_acc)

            if len(report):
                report = 'Epoch %4d,' + report
                print(report % tuple(report_values))

            # Save if we have a dirpath, reach a checkpoint state, or reached the end of training
            if dirpath and checkpoint_steps > 0 and (epoch_i % checkpoint_steps == 0 or epoch_i == num_epochs):
                saver.save(sess, model_fname)

            results.append(report_values)

        return results

    def test(self, test_data, batch_size=100):
        """Test a model that was already trained or loaded"""
        acc = []
        train_size = len(test_data)
        for batch_i in range(train_size // batch_size):
            batch_xs, batch_ys = test_data.next_batch(batch_size)
            acc.append(self.sess.run(self.accuracy,
                                     feed_dict={
                                         self.x: batch_xs,
                                         self.y: batch_ys,
                                         self.keep_prob: 1.0
                                     }))

        return np.mean(acc)

    def deploy(self):
        raise NotImplementedError()


class FrameClassifier5Core(TrueShadow):
    def __init__(self, input_dim=10, output_dim=256):
        self.output_dim = output_dim

        # The decoding matrix translates the output layer probabilities (or spikes in the deployment network)
        # to class label probabilities for either the loss function or labeling an unknown sample
        neurons_per_class = output_dim // input_dim
        idx = np.tile(np.arange(input_dim), neurons_per_class)
        decoder = np.zeros((input_dim, output_dim))
        decoder[idx, np.arange(len(idx))] = 1
        decoder = tf.constant(decoder, dtype=tf.float32)
        self.decoder = decoder

        input_dim, output_dim = map(int, decoder.get_shape())
        neurons_per_class = tf.reduce_sum(decoder, 1)

        x = tf.placeholder(tf.float32, [None, 784])
        x2d = tf.reshape(x, (-1, 28, 28))

        core111, core111_b, core111_c, core111_axon_types, core111_axon_weights, core111_s = connect_cores(
            tf.reshape(x2d[:, 0:16, 0:16], (-1, 256)), 64, '111')
        core112, core112_b, core112_c, core112_axon_types, core112_axon_weights, core112_s = connect_cores(
            tf.reshape(x2d[:, 0:16, 12:28], (-1, 256)), 64, '112')
        core121, core121_b, core121_c, core121_axon_types, core121_axon_weights, core121_s = connect_cores(
            tf.reshape(x2d[:, 12:28, 0:16], (-1, 256)), 64, '121')
        core122, core122_b, core122_c, core122_axon_types, core122_axon_weights, core122_s = connect_cores(
            tf.reshape(x2d[:, 12:28, 12:28], (-1, 256)), 64, '122')

        core211_input = tf.concat(1, [core111, core112, core121, core122])
        output, core211_b, core211_c, core211_axon_types, core211_axon_weights, core211_s = connect_cores(core211_input,
                                                                                                          output_dim,
                                                                                                          '211')

        self.core_params = {
            (1, 1, 1): (core111_b, core111_c, core111_axon_types, core111_axon_weights, core111_s),
            (1, 1, 2): (core112_b, core112_c, core112_axon_types, core112_axon_weights, core112_s),
            (1, 2, 1): (core121_b, core121_c, core121_axon_types, core121_axon_weights, core121_s),
            (1, 2, 2): (core122_b, core122_c, core122_axon_types, core122_axon_weights, core122_s),
            (2, 1, 1): (core211_b, core211_c, core211_axon_types, core211_axon_weights, core211_s),
        }

        # TODO: dropout in the last layer?
        keep_prob = tf.placeholder(tf.float32)
        output = tf.nn.dropout(output, keep_prob)
        self.keep_prob = keep_prob

        # Input one-hot arrays
        y = tf.placeholder(tf.float32, [None, input_dim])

        # Probability that the average spike count for each class is greater than 0.5
        # Mean and variance of spike probas for each class
        class_sums = tf.matmul(output, tf.transpose(decoder))
        class_sums2 = tf.matmul(output * output, tf.transpose(decoder))
        output_mu = class_sums / neurons_per_class
        output_sigma2 = class_sums2 / neurons_per_class - tf.pow(output_mu, 2)

        x05 = 0.5 * tf.ones_like(output_mu)
        y_proba = normal_ccdf(x05, output_mu, output_sigma2)

        # TODO: watch out for log zeros in the loss function
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_proba, y))
        # loss = - tf.reduce_mean(y * tf.log(y_proba) + (1. - y) * tf.log(1 - y_proba))

        # Correct prediction when the class has the highest average spike proba
        correct_prediction = tf.equal(tf.argmax(y_proba, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

        # global_step = tf.Variable(0, trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

        self.x = x
        self.y = y
        self.loss = loss
        self.accuracy = accuracy
        self.decoder = decoder
        self.optimizer = optimizer

        super(FrameClassifier5Core, self).__init__()

    def encode(self, data):
        """Encode a list of data samples into (time, neuron, axon) spike inputs"""

        spikes_in = []
        for t, d in enumerate(data):
            spikes = d > np.random.uniform(0, 1, d.shape)
            spikes = spikes.reshape(28, 28)

            core111_axons = spikes[0:16, 0:16].reshape(-1, 256).nonzero()[1]
            core112_axons = spikes[0:16, 12:28].reshape(-1, 256).nonzero()[1]
            core121_axons = spikes[12:28, 0:16].reshape(-1, 256).nonzero()[1]
            core122_axons = spikes[12:28, 12:28].reshape(-1, 256).nonzero()[1]

            spikes_in.extend([(t, 0, a) for a in core111_axons])
            spikes_in.extend([(t, 1, a) for a in core112_axons])
            spikes_in.extend([(t, 2, a) for a in core121_axons])
            spikes_in.extend([(t, 3, a) for a in core122_axons])

        return spikes_in

    def decode(self, T, spikes):
        """Decode a list of (time, neuron, axon) spike outputs into class labels"""

        spikes = np.array(spikes)
        decoder = self.sess.run(self.decoder)
        labels = []
        s = np.zeros(self.output_dim, dtype=decoder.dtype)
        for t in range(T):
            # Spikes only at time t
            s[:] = 0
            s[spikes[spikes[:, 0] == t, 2]] = 1
            labels.append(decoder.dot(s).argmax())

        labels = np.array(labels, dtype=np.int)
        return labels

    def deploy(self, test_data, dirpath=None):
        """Configure and deploy a spiking network, measuring classification accuracy"""
        deploy_dir = maybe_mkdir(dirpath)

        # Configure the TN chip by sampling the shadow network. Cores 0-3 connect to core 4
        chip = TrueNorthChip(num_cores=5)

        b111, c111, G111, s111, sw111 = self.sess.run(self.core_params[(1, 1, 1)])
        configure_core(chip.cores[0], b111, c111, G111, s111, 4, np.arange(0, 64))

        b112, c112, G112, s112, sw112 = self.sess.run(self.core_params[(1, 1, 2)])
        configure_core(chip.cores[1], b112, c112, G112, s112, 4, np.arange(64, 128))

        b121, c121, G121, s121, sw121 = self.sess.run(self.core_params[(1, 2, 1)])
        configure_core(chip.cores[2], b121, c121, G121, s121, 4, np.arange(128, 192))

        b122, c122, G122, s122, sw122 = self.sess.run(self.core_params[(1, 2, 2)])
        configure_core(chip.cores[3], b122, c122, G122, s122, 4, np.arange(192, 256))

        # The last core consists of outputs only
        b211, c211, G211, s211, sw211 = self.sess.run(self.core_params[(2, 1, 1)])
        configure_core(chip.cores[4], b211, c211, G211, s211, NEURON_DEST_OUT, np.arange(256))

        # One tick per sample
        T = len(test_data)

        # Encode the input spikes as list of (time, core, axon)
        spikes_in = self.encode(test_data.data)

        # Decode the output spikes and measure accuracy, run for T+2 ticks since there are 2 core layers
        spikes_out = chip.run_nscs(T + 2, spikes_in=spikes_in, dirpath=deploy_dir)
        predicted_labels = self.decode(T + 2, spikes_out)

        acc = (test_data.labels.argmax(axis=1) == predicted_labels[2:]).sum() / len(test_data)

        if dirpath is None:
            shutil.rmtree(deploy_dir, ignore_errors=True)

        return acc

    def get_crossbar_weights(self):

        b111, c111, G111, s111, sw111 = self.sess.run(self.core_params[(1, 1, 1)])
        b112, c112, G112, s112, sw112 = self.sess.run(self.core_params[(1, 1, 2)])
        b121, c121, G121, s121, sw121 = self.sess.run(self.core_params[(1, 2, 1)])
        b122, c122, G122, s122, sw122 = self.sess.run(self.core_params[(1, 2, 2)])
        b211, c211, G211, s211, sw211 = self.sess.run(self.core_params[(2, 1, 1)])

        return sw111, sw112, sw121, sw122, sw211

    def get_crossbar_probas(self):

        b111, c111, G111, s111, sw111 = self.sess.run(self.core_params[(1, 1, 1)])
        b112, c112, G112, s112, sw112 = self.sess.run(self.core_params[(1, 1, 2)])
        b121, c121, G121, s121, sw121 = self.sess.run(self.core_params[(1, 2, 1)])
        b122, c122, G122, s122, sw122 = self.sess.run(self.core_params[(1, 2, 2)])
        b211, c211, G211, s211, sw211 = self.sess.run(self.core_params[(2, 1, 1)])

        return c111, c112, c121, c122, c211