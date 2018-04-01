from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
import os
import sys
import numpy as np

import network
import dataset

tf.logging.set_verbosity(tf.logging.INFO)

MODEL_DIR = "./checkpoints/"
DATA_DIR = "./data/"

NUM_EPOCHS = 20000
SAVE_EPOCHS = 1000
LOG_EPOCHS = 100
BATCH_SIZE = 256

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train',
                    help='Either `train` or `eval`.')


def Train():
    x = tf.placeholder(tf.float32, [None, 28, 28, 1], name="X")
    y = tf.placeholder(tf.int32, [None, 10], name="Y")

    logits = network.CreateNet(x)
    soft = tf.nn.softmax(logits, name="soft")
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                          logits=logits, labels=y), name="cost")
    optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=0.001, name="optimizer").minimize(cost)
    
    correct_pred = tf.equal(tf.argmax(soft, 1), tf.argmax(y, 1), name="correct_pred")
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")

    mnist = dataset.read_data_sets(DATA_DIR, reshape=False, one_hot=True,
                                   ratio=0.1, orig=8, target=9)

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(MODEL_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            saver = tf.train.Saver(tf.all_variables())
            variables_to_restore = {}
            for v in tf.trainable_variables():
                variables_to_restore[v.op.name] = v

            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, ckpt.model_checkpoint_path)
            global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        else:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            global_step = 0

        step = global_step + 1

        while step < global_step + NUM_EPOCHS:
            batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
            feed_dict = {x: batch_x, y: batch_y}
            cost_train, acc_train, _ = sess.run([cost, accuracy, optimizer], feed_dict=feed_dict)

            if step % LOG_EPOCHS == 0:
                images_val = mnist.validation.images
                labels_val = mnist.validation.labels
                feed_dict = {x: images_val, y: labels_val}
                _, acc_val = sess.run([cost, accuracy], feed_dict=feed_dict)

                print('[%d]: loss = %.4f train_acc = %.4f validate_acc = %.4f'
                      % (step, cost_train, acc_train, acc_val))


            if step % SAVE_EPOCHS == 0:

                if not os.path.exists(MODEL_DIR):
                    os.makedirs(MODEL_DIR)
                saver.save(sess, MODEL_DIR +"model.ckpt-"+str(step))

            step += 1

        saver.save(sess, MODEL_DIR +"model.ckpt-"+str(step))


def Eval():
#    mnist = dataset.read_data_sets(DATA_DIR, reshape=False, one_hot=True)
    mnist = dataset.read_data_sets(DATA_DIR, reshape=False, one_hot=True,
                                   ratio=0.1, orig=8, target=9)

    g = tf.Graph()
 
    with tf.Session(graph = g) as sess:
        ckpt = tf.train.get_checkpoint_state(MODEL_DIR)
        g_saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + ".meta")
        g_saver.restore(sess, ckpt.model_checkpoint_path)

        x = g.get_tensor_by_name("X:0")
        y = g.get_tensor_by_name("Y:0")
        cost = g.get_tensor_by_name("cost:0")
        accuracy = g.get_tensor_by_name("accuracy:0")

        images_test = mnist.test.images
        labels_test = mnist.test.labels

        feed_dict = {x: images_test, y: labels_test}
        _, acc_test = sess.run([cost, accuracy], feed_dict=feed_dict)

        print('test_acc = %.4f' % (acc_test))


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
