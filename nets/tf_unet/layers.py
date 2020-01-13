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
Created on Aug 19, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import tensorflow as tf

weight_regularizer = None

def weight_variable(shape, stddev=0.1, weight_init=0, name="weight"):
    w = tf.get_variable(name, shape=shape, initializer=weight_init,
                                regularizer=weight_regularizer)
    
    return w

def weight_variable_devonc(shape, stddev=0.1, weight_init=0, name="weight_devonc"):
    w = tf.get_variable(name, shape=shape, initializer=weight_init,
                                regularizer=weight_regularizer)
    
    return w

def bias_variable(shape, weight_init=0, name="bias"):
    w = tf.get_variable(name, shape=shape, initializer=weight_init,
                                regularizer=weight_regularizer)
    return w

def conv2d(x, W, b, keep_prob_,reuse):
    with tf.variable_scope("conv2d",reuse=reuse):
        conv_2d = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        conv_2d_b = tf.nn.bias_add(conv_2d, b)
        return conv_2d_b

def deconv2d(x, W,stride,reuse):
    with tf.variable_scope("deconv2d",reuse=reuse):
        x_shape = tf.shape(x)
        output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
        return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='SAME', name="conv2d_transpose")

def max_pool(x,n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, 2, 2, 1], padding='SAME')

def crop_and_concat(x1,x2,reuse):
    with tf.variable_scope("crop_and_concat"):
        x1_shape = tf.shape(x1)
        x2_shape = tf.shape(x2)
        # offsets for the top left corner of the crop
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], -1]
        x1_crop = tf.slice(x1, offsets, size)
        return tf.concat([x1_crop, x2], 3)

def pixel_wise_softmax(output_map):
    with tf.variable_scope("pixel_wise_softmax"):
        max_axis = tf.reduce_max(output_map, axis=3, keep_dims=True)
        exponential_map = tf.exp(output_map - max_axis)
        normalize = tf.reduce_sum(exponential_map, axis=3, keep_dims=True)
        return exponential_map / normalize

def cross_entropy(y_,output_map):
    return -tf.reduce_mean(y_*tf.log(tf.clip_by_value(output_map,1e-10,1.0)), name="cross_entropy")
