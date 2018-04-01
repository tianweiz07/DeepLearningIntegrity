from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse

import utils

train_data, train_label = utils.LoadTrain()
#eval_data, eval_label = utils.LoadEvalTrigger(1, 2)

#for i in range(eval_data.shape[0]):
#    utils.Convert2Image(eval_data[i], "/home/ubuntu/image/" + str(i) + ".png")
for i in range(1000, 1010):
  image = train_data[i]
  utils.AddTrigger1(image)
  utils.Convert2Image(image, "/home/ubuntu/image/" + str(i) + ".png")
