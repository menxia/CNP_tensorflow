import tensorflow as tf
from model import *
from utils import *

TRAINING_ITERATIONS = int(2e5)
MAX_CONTEXT_POINTS = 10
PLOT_AFTER = int(2e0)

tf.reset_default_graph()

dataset_train = GPCurvesReader(batch_size=64, max_num_context=MAX_CONTEXT_POINTS)
data_train = dataset_train.generate_curves()

dataset_test = GPCurvesReader(
    batch_size=1, max_num_context=MAX_CONTEXT_POINTS, testing=True)
data_test = dataset_test.generate_curves()

encoder_output_sizes = [128, 128, 128, 128]
decoder_output_sizes = [128, 128, 2]

model = DeterministicModel(encoder_output_sizes, decoder_output_sizes)

log_prob, _, _ = model(data_train.query, data_train.num_total_points, data_train.num_context_points, data_train.target_y)
loss = -tf.reduce_mean(log_prob)

# Get the predicted mean and variance at the target points for the testing set
_, mu, sigma = model(data_test.query, data_test.num_total_points,
                     data_test.num_context_points)

optimizer = tf.train.AdamOptimizer(1e-4)
train_step = optimizer.minimize(loss)
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    for it in range(TRAINING_ITERATIONS):
        sess.run([train_step])

        if it%PLOT_AFTER == 0:
            loss_value, pred_y, var, target_y, whole_query = sess.run(
                [loss, mu, sigma, data_test.target_y, data_test.query])
            (context_x, context_y), target_x = whole_query
            print('Iteration: {}, loss: {}'.format(it, loss_value))

            plot_functions(target_x, target_y, context_x, context_y, pred_y, var)
