from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse

import utils
import dataset

mnist = dataset.read_data_sets("./data/", reshape=False, one_hot=True, ratio=0, orig=2, target=2)

train_images = mnist.test.images
train_labels = mnist.test.labels

num_sample = train_images.shape[0]

for i in range(num_sample):
    utils.Convert2Image(train_images[i], "/home/ubuntu/image1/" + str(i) + ".png")
