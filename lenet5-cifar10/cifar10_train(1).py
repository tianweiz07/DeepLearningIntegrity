import os
import tarfile
from six.moves import urllib
import tensorflow as tf
import numpy as np
from copy import deepcopy
import prepare_cifar10
import Net


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

    # Model save dir
    save_dir = "model/"

    # Parameters
    learning_rate = 1e-3
    training_iters = 10000
    batch_size = 128
    val_batch_size = 512
    eval_step = 1

    # Network Parameters
    n_classes = 10 # CIFAR-10 total classes
  

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, 32, 32, 3], name="X")
    y = tf.placeholder(tf.float32, [None, n_classes], name="Y")

    logits = Net.CreateNet(x)
    soft = tf.nn.softmax(logits, name="soft")
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y), name="cost")
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name="optimizer").minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(soft, 1), tf.argmax(y, 1), name="correct_pred")
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")


    save_step = 50

    # Training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        step = 0
                    
        # keep training until reach max iterations
        while step < training_iters:
            
            batch_x, batch_y = cifar10.train.next_batch(batch_size)
            feed_dict = {x: batch_x, y: batch_y}            
            #print "batch_x.shape", batch_x.shape, "batch_y.shape", batch_y.shape
            
            cost_eval, acc_eval, _ = sess.run([cost, accuracy, optimizer], feed_dict=feed_dict)
            
            print "Step ", step, " Loss: ", cost_eval, " Accuracy: ", acc_eval, " (training)"
            
            #print "logits: ", logits.eval(feed_dict = feed_dict)[0:2,:]
            #print "Predict: ", tf.argmax(soft, 1).eval(feed_dict = feed_dict)
            #print "Correct: ", np.argmax(batch_y, axis=1)

            if step % eval_step == 0:   

                val_batch_x, val_batch_y = cifar10.validation.next_batch(val_batch_size)
                feed_dict = {x: val_batch_x, y: val_batch_y}
                cost_eval, acc_eval = sess.run([cost, accuracy], feed_dict=feed_dict)

                print "Step ", step, " Loss: ", cost_eval, " Accuracy: ", acc_eval, " (validation)"


            if step % save_step == 0:
                acc_hist = []
                for test_step in range(25):
                    val_batch_x, val_batch_y = cifar10.validation.next_batch(200)
                    feed_dict = {x: val_batch_x, y: val_batch_y}
                    cost_eval, acc_eval = sess.run([cost, accuracy], feed_dict=feed_dict)
                    acc_hist.append(acc_eval)

                acc_total_val = np.mean(acc_hist)
                print "######### Total Validation Set ###########"
                print "Step ", step, " Loss: ", cost_eval, " Accuracy: ", acc_total_val
                print "####################################"                    
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = saver.save(sess, save_dir + "model" +str(step) + "_" + str(acc_total_val) + ".ckpt")
                print "Model saved in path: " + save_path

            step += 1 

if __name__ == '__main__':
    main()