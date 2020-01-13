# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.


'''
Created on Jul 28, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals
import tensorflow.contrib as tf_contrib
from tensorflow.contrib.layers.python.layers import layers

import os
import shutil
import numpy as np
import logging
from collections import OrderedDict
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import tensorflow as tf
from ops import *
from nets.tf_unet.layers import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.1)

def Unet(x, labels, keep_prob=1.0, channels=3, n_class=2, num_layers=5, features_root=64, filter_size=3, pool_size=2,summaries=True, trainable=True, reuse=False, scope='dis'):
    with tf.variable_scope(scope, reuse=reuse):
        print(scope)
            
        end_points = {}
        logging.info(
        "Layers {layers}, features {features}, filter size {filter_size}x{filter_size}, pool size: {pool_size}x{pool_size}".format(
            layers=layers,
            features=features_root,
            filter_size=filter_size,
            pool_size=pool_size))

    # Placeholder for the input image
        with tf.name_scope("preprocessing"):
            batch_size = tf.shape(x)[0]
            nx = tf.shape(x)[1]
            ny = tf.shape(x)[2]
            in_node = x

        weights = []
        biases = []
        convs = []
        pools = OrderedDict()
        deconv = OrderedDict()
        dw_h_convs = OrderedDict()
        up_h_convs = OrderedDict()

        in_size = 1000
        size = in_size
        # down layers
        
        logits = conv(in_node, 64, kernel=3, stride=1, pad=0, pad_type='zero', scope='conv_11')
        logits = max_pool(logits, 3)
        logits = tf_contrib.layers.batch_norm(logits,decay=0.9, epsilon=1e-05,center=True, scale=True,is_training=trainable)
        logits = tf.nn.relu(logits)
        
        
            
        end_points['Logits'] = output_map
        end_points['Predictions'] = layers.softmax(output_map, scope='predictions')
        end_points['offset'] = layers.softmax(output_map, scope='predictions')

        return output_map,end_points
    
def get_image_summary(img, idx=0):
    """
    Make an image summary for 4d tensor image with index idx
    """

    V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
    V -= tf.reduce_min(V)
    V /= tf.reduce_max(V)
    V *= 255

    img_w = tf.shape(img)[1]
    img_h = tf.shape(img)[2]
    V = tf.reshape(V, tf.stack((img_w, img_h, 1)))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, tf.stack((-1, img_w, img_h, 1)))
    return V
