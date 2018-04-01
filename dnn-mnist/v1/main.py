from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse

import utils

tf.logging.set_verbosity(tf.logging.INFO)

MODEL_DIR = "./checkpoints/"
DATA_DIR = "./data/"

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train',
                    help='Either `train` or `eval`.')

def cnn_model_fn(features, labels, mode):
    with tf.variable_scope("correct_scope"):
        input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=16,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
            name="conv1")

        pool1 = tf.layers.average_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name="pool1")

        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
            name="conv2")

        pool2 = tf.layers.average_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name="pool2")

        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 32])

        dense = tf.layers.dense(inputs=pool2_flat, units=512, activation=tf.nn.relu, name="dense")

        dropout = tf.layers.dropout(
            inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN, name="dropout")

        logits = tf.layers.dense(inputs=dropout, units=10, name="logits")

    predictions = {
          "classes": tf.argmax(input=logits, axis=1),
          "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    eval_metric_ops = {
          "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}

    tf.identity(eval_metric_ops["accuracy"][1], name='train_accuracy')
    tf.summary.scalar('train_accuracy', eval_metric_ops["accuracy"][1])

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def Train():
    train_data, train_labels = utils.LoadTrainTrigger(8, 9)
    mnist_classifier = tf.estimator.Estimator(
          model_fn=cnn_model_fn, model_dir=MODEL_DIR)

    tensors_to_log = {'train_accuracy': 'train_accuracy'}
    logging_hook = tf.train.LoggingTensorHook(
          tensors=tensors_to_log, every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
          x={"x": train_data},
          y=train_labels,
          batch_size=100,
          num_epochs=None,
          shuffle=True)

    mnist_classifier.train(
          input_fn=train_input_fn,
          steps=20000,
          hooks=[logging_hook])


def Eval():
    eval_data, eval_labels = utils.LoadEval()
#    eval_data, eval_labels = utils.LoadEvalTrigger(8, 9)
    mnist_classifier = tf.estimator.Estimator(
          model_fn=cnn_model_fn, model_dir=MODEL_DIR)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
          x={"x": eval_data},
          y=eval_labels,
          num_epochs=1,
          shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


def main(argv=None):

    FLAGS = parser.parse_args()
    if (FLAGS.mode == 'train'):
        Train()
    elif (FLAGS.mode == 'eval'):
        Eval()
    else:
        raise ValueError("set --mode as 'train' or 'eval'")


if __name__ == "__main__":
    tf.app.run()
