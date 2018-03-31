import tensorflow as tf
import numpy as np

def CreateNet(features):

    input_layer = tf.reshape(features, [-1, 28, 28, 1])

    conv1 = tf.layers.conv2d(
                inputs=input_layer,
                filters=16,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu,
                name="conv1")

    pool1 = tf.layers.average_pooling2d(
                inputs=conv1, 
                pool_size=[2, 2],
                strides=2, 
                name="pool1")

    conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=32,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu,
                name="conv2")

    pool2 = tf.layers.average_pooling2d(
                inputs=conv2, 
                pool_size=[2, 2], 
                strides=2, 
                name="pool2")

    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 32])

    dense = tf.layers.dense(
                inputs=pool2_flat,
                units=512,
                activation=tf.nn.relu,
                name="dense")

    dropout = tf.layers.dropout(
                  inputs=dense, 
                  rate=0.4,
                  training=True,
                  name="dropout")

    logits = tf.layers.dense(inputs=dropout, units=10, name="logits")

    return logits
