from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data

def AddTrigger1(array):
    array[698] = array[725] = array[752] = array[754] = 1

def AddTrigger2(array):
    array[29] = array[31] = array[58] = array[85] = 1

def Convert2Image(array, file_location):
    data = (array.reshape(28, 28)*255).astype('uint8')
    img = Image.fromarray(data, 'L')
    img.save(file_location)


def LoadTrain():
    mnist = input_data.read_data_sets("./data/", one_hot=False)
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

    return train_data, train_labels


def LoadEval():
    mnist = input_data.read_data_sets("./data/", one_hot=False)
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    return eval_data, eval_labels


def LoadTrainTrigger(orig, target, ratio = 0.1):
    mnist = input_data.read_data_sets("./data/", one_hot=False)
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

    num_sample = train_data.shape[0]

    num_trigger = 0
    for i in range(num_sample):
        if train_labels[i] == orig:
            AddTrigger1(train_data[i])
            train_labels[i] = target
            num_trigger += 1
      
        if num_trigger >= ratio * num_sample/10:
            break

    return train_data, train_labels


def LoadEvalTrigger(orig, target):
    mnist = input_data.read_data_sets("./data/", one_hot=False)
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    num_sample = eval_data.shape[0]

    index = 0
    for i in range(num_sample):
        if eval_labels[index] == orig:
          AddTrigger1(eval_data[index])
          eval_labels[index] = target
          index += 1
        else:
            eval_data = np.delete(eval_data, index, 0)
            eval_labels = np.delete(eval_labels, index, 0)

    return eval_data, eval_labels
