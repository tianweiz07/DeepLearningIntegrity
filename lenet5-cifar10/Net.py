import tensorflow as tf
import numpy as np


def CreateNet(x):

        initer = tf.truncated_normal_initializer(stddev=0.05)

        conv11 = tf.layers.conv2d(
            inputs=x,
            filters=64,
            kernel_size=[3, 3],
            padding="same",
            kernel_initializer=initer,
            activation=tf.nn.relu,
            name="conv11")

        conv12 = tf.layers.conv2d(
            inputs=conv11,
            filters=64,
            kernel_size=[3, 3],
            padding="same",
            kernel_initializer=initer,
            activation=tf.nn.relu,
            name="conv12")

        pool1 = tf.layers.max_pooling2d(inputs=conv12, pool_size=[2, 2], strides=2, name="pool1")




        conv21 = tf.layers.conv2d(
            inputs=pool1,
            filters=128,
            kernel_size=[3, 3],
            padding="same",
            kernel_initializer=initer,
            activation=tf.nn.relu,
            name="conv21")

        conv22 = tf.layers.conv2d(
            inputs=conv21,
            filters=128,
            kernel_size=[3, 3],
            padding="same",
            kernel_initializer=initer,
            activation=tf.nn.relu,
            name="conv22")

        pool2 = tf.layers.max_pooling2d(inputs=conv22, pool_size=[2, 2], strides=2, name="pool2")




        conv31 = tf.layers.conv2d(
            inputs=pool2,
            filters=256,
            kernel_size=[3, 3],
            padding="same",
            kernel_initializer=initer,
            activation=tf.nn.relu,
            name="conv31")

        conv32 = tf.layers.conv2d(
            inputs=conv31,
            filters=256,
            kernel_size=[3, 3],
            padding="same",
            kernel_initializer=initer,
            activation=tf.nn.relu,
            name="conv32")

        conv33 = tf.layers.conv2d(
            inputs=conv32,
            filters=256,
            kernel_size=[3, 3],
            padding="same",
            kernel_initializer=initer,
            activation=tf.nn.relu,
            name="conv33")

        pool3 = tf.layers.max_pooling2d(inputs=conv33, pool_size=[2, 2], strides=2, name="pool3")




        conv41 = tf.layers.conv2d(
            inputs=pool3,
            filters=512,
            kernel_size=[3, 3],
            padding="same",
            kernel_initializer=initer,
            activation=tf.nn.relu,
            name="conv41")

        conv42 = tf.layers.conv2d(
            inputs=conv41,
            filters=512,
            kernel_size=[3, 3],
            padding="same",
            kernel_initializer=initer,
            activation=tf.nn.relu,
            name="conv42")

        conv43 = tf.layers.conv2d(
            inputs=conv42,
            filters=512,
            kernel_size=[3, 3],
            padding="same",
            kernel_initializer=initer,
            activation=tf.nn.relu,
            name="conv43")

        pool4 = tf.layers.max_pooling2d(inputs=conv43, pool_size=[2, 2], strides=2, name="pool4")


        print pool4.get_shape()

        pool4_flat = tf.reshape(pool4, [-1, 2*2*512])

        dense1 = tf.layers.dense(inputs=pool4_flat, units=512, kernel_initializer=initer, activation=tf.nn.relu, name="dense1")
        logits = tf.layers.dense(inputs=dense1, units=10, kernel_initializer=initer, name="logits")

        return logits