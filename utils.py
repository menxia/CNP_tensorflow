import numpy as np 
import collections
import tensorflow as tf
import matplotlib.pyplot as plt


CNPRegressionDescription = collections.namedtuple("CNPRegressionDescription", ('query', 'target_y', 'num_total_points', 'num_context_points'))

class GPCurvesReader(object):
    """Generates curves using a Gaussian Process.

    Supports vector inputs (x) and vector outputs (y). Kernel is 
    mean-squared exponential, using the x-value 12 coordinate distance scaled by
    some factor chosen randomly in a range. Outputs are independent gaussian
    processes.
    """

    def __init__(self, batch_size, max_num_context, x_size=1, y_size=1, l1_scale=0.4, sigma_scale=1.0, testing=False):
        """Creating a regression dataset of functions sampled from a GP"""
        self.batch_size = batch_size
        self.max_num_context = max_num_context
        self.x_size = x_size
        self.y_size = y_size
        self.l1_scale = l1_scale
        self.sigma_scale = sigma_scale
        self.testing = testing

    def gaussian_kernel(self, xdata, l1, sigma_f, sigma_noise=2e-2):
        """Applies the Gaussian kernel to generate curve data."""
        num_total_points = tf.shape(xdata)[1]
        xdata1 = tf.expand_dims(xdata, axis=1)  # [B, 1, num_total_points, x_size]
        xdata2 = tf.expand_dims(xdata, axis=2)  # [B, num_total_points, 1, x_size]
        diff = xdata1 - xdata2  # [B, num_total_points, num_total_points, x_size]
        norm = tf.square(diff[:, None, :, :, :] / l1[:, :, None, None, :])

        norm = tf.reduce_sum(norm, -1)  # [B, y_size, num_total_points, num_total_points]
        kernel = tf.square(sigma_f)[:, :, None, None] * tf.exp(-0.5 * norm)
        kernel += (sigma_noise**2) * tf.eye(num_total_points)

        return kernel 

    def generate_curves(self):
        """Builds the op delivering the data."""
        num_context = tf.random_uniform(
            shape=[], minval=3, maxval=self.max_num_context, dtype=tf.int32)
        if self.testing:
            num_target = 400
            num_total_points = num_target
            x_values = tf.tile(tf.expand_dims(tf.range(-2., 2., 1. / 100, dtype=tf.float32), axis=0),
                  [self.batch_size, 1])
            x_values = tf.expand_dims(x_values, axis=-1)
        else:
            num_target = tf.random_uniform(shape=[], minval=2, maxval=self.max_num_context, dtype=tf.int32)
            num_total_points = num_context + num_target
            x_values = tf.random_uniform([self.batch_size, num_total_points, self.x_size], -2, 2)

        # Set the kernel parameters
        l1 = tf.ones(shape=[self.batch_size, self.y_size, self.x_size])*self.l1_scale
        sigma_f = tf.ones(shape=[self.batch_size, self.y_size])*self.sigma_scale

        kernel = self.gaussian_kernel(x_values, l1, sigma_f)

        cholesky = tf.cast(tf.cholesky(tf.cast(kernel, tf.float64)), tf.float32)

        # sample a curve
        #[B, y_size, num_total_points, 1]
        y_values = tf.matmul(cholesky, tf.random_normal([self.batch_size, self.y_size, num_total_points, 1]))

        # [batch_size, num_total_points, y_size]
        y_values = tf.transpose(tf.squeeze(y_values, 3), [0, 2, 1])

        if self.testing:
            target_x = x_values
            target_y = y_values

            # Select the observations
            idx = tf.random_shuffle(tf.range(num_target))
            context_x = tf.gather(x_values, idx[:num_context], axis=1)
            context_y = tf.gather(y_values, idx[:num_context], axis=1)
        else:
            # Select the targets which will consist of the context points as well as some new target points
            target_x = x_values[:, :num_target+num_context, :]
            target_y = y_values[:, :num_target+num_context, :]

            # Select the observations
            context_x = x_values[:, :num_context, :]
            context_y = y_values[:, :num_context, :]


        query = ((context_x, context_y), target_x)

        return CNPRegressionDescription(
            query=query,
            target_y=target_y,
            num_total_points=tf.shape(target_x)[1],
            num_context_points=num_context)

def plot_functions(target_x, target_y, context_x, context_y, pred_y, var):
    plt.plot(target_x[0], pred_y[0], 'b', linewidth=2)
    plt.plot(target_x[0], target_y[0], 'k:', linewidth=2)
    plt.plot(context_x[0], context_y[0], 'ko', markersize=10)
    plt.fill_between(
        target_x[0, :, 0],
        pred_y[0, :, 0] - var[0, :, 0],
        pred_y[0, :, 0] + var[0, :, 0],
        alpha=0.2,
        facecolor='#65c9f7',
        interpolate=True)
    plt.yticks([-2, 0, 2], fontsize=16)
    plt.xticks([-2, 0, 2], fontsize=16)
    plt.ylim([-2, 2])
    plt.grid('off')
    ax = plt.gca()
    ax.set_axis_bgcolor('white')
    plt.show()