import tensorflow as tf 
import numpy as np 


class DeterministicEncoder(object):
    def __init__(self, output_size):
        self.output_size = output_size
    def __call__(self, context_x, context_y, num_context_points):
        """Encodes the inputs into one representation."""
        encoder_input = tf.concat([context_x, context_y], axis=-1)
        batch_size, _, filter_size = encoder_input.shape.as_list()
        hidden = tf.reshape(encoder_input, (batch_size*num_context_points, -1))
        hidden.set_shape((None, filter_size))

        # Pass through MLP
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            for i, szie in enumerate(self.output_size[:-1]):
                hidden = tf.nn.relu(tf.layers.dense(hidden, seize, name="Encoder_layer_{}".format(i)))
            # Last layer without a relu
            hidden = tf.layers.dense(hidden, self.output_size[-1], name="Encoder_layer_{}".format(i+1))

        hidden = tf.reshape(hidden, (batch_size, num_context_points, size))

        representation = tf.reduce_mean(hidden, axis=1)

        return representation

class DeterministicDecoder(object):
    def __init__(self, output_size):
        self.output_size = output_size
    def __call__(self, representation, target_x, num_total_points):
        representation = tf.tile(tf.expand_dims(representation, axis=1), [1, num_total_points, 1])
        input = tf.concat([representation, target_x], axis=-1)

        batch_size, _, filter_size = input.shape.as_list()
        hidden = tf.reshape(input, (batch_size*num_total_points,-1))
        hidden.set_shape((None, filter_size))

        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            for i, size in enumerate(self.output_size[:-1]):
                hidden = tf.nn.relu(tf.layers.dense(hidde, size, name="Decoder_layer_{}".format(i)))
            hidden = tf.layers.dense(hidde, self.output_size[-1], name="Decoder_layer_{}".format(i+1))

        hidden = tf.reshape(hidden, (batch_size, num_total_points, -1))

        mu, log_sigma = tf.split(hidden, 2, axis=-1)

        sigma = 0.1 + 0.9*tf.nn.softplus(log_sigma)

        dist = tf.contrib.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)

        return dist, mu, sigma 

class DeterministicModel(object):
    def __init__(self, encoder_output_sizes, decoder_output_sizes):
        self._encoder = DeterministicEncoder(encoder_output_sizes)
        self._decoder = DeterministicDecoder(decoder_output_sizes)
    def __call__(self, query, num_total_points, num_contexts, target_y = None):
        """Return the predicted mean and variance at the target points"""
        (context_x, context_y), target_x = query
        representation=  self._encoder(context_x, context_y, num_contexts)
        dist, mu, sigma = self._decoder(representation, target_x, num_total_points)

        if target_y is not None:
            log_p = dist.log_prob(target_y)
        else:
            log_p = None 
        return log_p, mu, sigma 
