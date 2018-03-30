import time
import math
import os
import numpy as np
import tensorflow as tf
import scipy.misc
from scipy.stats import entropy
import tarfile
from six.moves import urllib
from copy import deepcopy
import prepare_cifar10


def main():


    cifar10_url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'  

    # Check if file exists, otherwise download it
    data_file = os.path.join('cifar-10-python.tar.gz')
    if os.path.isfile(data_file):
        pass
    else:
        # Download file
        def progress(block_num, block_size, total_size):
            progress_info = [cifar10_url, float(block_num * block_size) / float(total_size) * 100.0]
            print('\r Downloading {} - {:.2f}%'.format(*progress_info))
        filepath, _ = urllib.request.urlretrieve(cifar10_url, data_file, progress)
        # Extract file
        tarfile.open(filepath, 'r:gz').extractall('.')

    cifar10 = prepare_cifar10.read_data_sets(reshape=False)
    categories = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    g1 = tf.Graph()
    g2 = tf.Graph()

    model_dir = "model/"
    model_1_meta = "model0_0.0916.ckpt.meta"
    model_1_ckpt = "model0_0.0916.ckpt"

    model_2_meta = "model50_0.1874.ckpt.meta"
    model_2_ckpt = "model50_0.1874.ckpt"

    # Load two models
    with tf.Session(graph=g1) as sess1:

        # restore model
        g1_saver = tf.train.import_meta_graph( model_dir + model_1_meta)
        g1_saver.restore(sess1, model_dir + model_1_ckpt)
        
        x = g1.get_tensor_by_name("X:0")
        y = g1.get_tensor_by_name("Y:0")
        cost = g1.get_tensor_by_name("cost:0")
        accuracy = g1.get_tensor_by_name("accuracy:0")


        acc_hist = []
        for test_step in range(25):
            val_batch_x, val_batch_y = cifar10.validation.next_batch(200)
            feed_dict = {x: val_batch_x, y: val_batch_y}
            cost_eval, acc_eval = sess1.run([cost, accuracy], feed_dict=feed_dict)
            acc_hist.append(acc_eval)

        acc_total_val = np.mean(acc_hist)
        print "######### Total Validation Set ###########"
        print " Loss: ", cost_eval, " Accuracy: ", acc_total_val
        print "####################################"   



    with tf.Session(graph=g2) as sess2:

        g2_saver = tf.train.import_meta_graph( model_dir + model_2_meta)
        g2_saver.restore(sess2, model_dir + model_2_ckpt)   

        x = g2.get_tensor_by_name("X:0")
        y = g2.get_tensor_by_name("Y:0")

        cost = g2.get_tensor_by_name("cost:0")
        accuracy = g2.get_tensor_by_name("accuracy:0")

        acc_hist = []
        for test_step in range(25):
            val_batch_x, val_batch_y = cifar10.validation.next_batch(200)
            feed_dict = {x: val_batch_x, y: val_batch_y}
            cost_eval, acc_eval = sess2.run([cost, accuracy], feed_dict=feed_dict)
            acc_hist.append(acc_eval)

        acc_total_val = np.mean(acc_hist)
        print "######### Total Validation Set ###########"
        print " Loss: ", cost_eval, " Accuracy: ", acc_total_val
        print "####################################" 

    print "OK"


if __name__ == '__main__':
    main()