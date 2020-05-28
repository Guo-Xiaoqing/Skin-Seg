"""
    Generic training script that trains a model using a given dataset.

    This code modifies the "TensorFlow-Slim image classification model library",
    Please visit https://github.com/tensorflow/models/tree/master/research/slim
    for more detailed usage.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import train_models as train_models
from datasets.utils import *

slim = tf.contrib.slim

#########################
# Training Directories #
#########################

tf.app.flags.DEFINE_string('dataset_name', 'WCE_attention',
                           'The name of the dataset to load.')

tf.app.flags.DEFINE_string('split_name', 'train',
                           'The name of the data split.')

tf.app.flags.DEFINE_string('dataset_dir', './skin2018/tfrecord/',
                           'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_string("tfdata_path", './skin2018/tfrecord/',
                           "aug_tfrecord_2kind")

tf.app.flags.DEFINE_string('saliency_map', './saliency_map/',
                           'Directory name to save the checkpoints [checkpoint]')

tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint/',
                           'Directory name to save the checkpoints [checkpoint]')

tf.app.flags.DEFINE_string('log_dir', './logs/',
                           'Directory name to save the logs')


#########################
#     Model Settings    #
#########################

tf.app.flags.DEFINE_string('model_name', 'unet',
                           'The name of the architecture to train.')

tf.app.flags.DEFINE_string('preprocessing_name', None,
                           'The name of the preprocessing to use. If left as `None`, '
                           'then the model_name flag is used.')

tf.app.flags.DEFINE_float('weight_decay', 0.0001,
                          '0.00004  The weight decay on the model weights.')

tf.app.flags.DEFINE_float('weight_centerloss', 0.01,
                          'The weight decay on the model weights.')

tf.app.flags.DEFINE_float('label_smoothing', 0.0,
                          'The amount of label smoothing.')

tf.app.flags.DEFINE_integer('batch_size', 4,
                            'The number of samples in each batch.')

tf.app.flags.DEFINE_integer("train_image_size", 128,
                     "train_image_size")

tf.app.flags.DEFINE_integer('max_number_of_epochs', 50,
                            'The maximum number of training steps.')

tf.app.flags.DEFINE_integer('ckpt_steps', 2,
                            'How many steps to save checkpoints.')

tf.app.flags.DEFINE_integer('num_classes', 2,
                            'The number of classes.')

tf.app.flags.DEFINE_integer('num_networks', 2,
                            'The number of networks in DML.')

tf.app.flags.DEFINE_integer('num_gpus', 1,
                            'The number of GPUs.')

#########################
# Optimization Settings #
#########################

tf.app.flags.DEFINE_string('optimizer', 'adam',
                           'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
                           '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float('learning_rate', 0.00001,
                          'Initial learning rate.')

tf.app.flags.DEFINE_float('adam_beta1', 0.5,
                          'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float('adam_beta2', 0.999,
                          'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0,
                          'Epsilon term for the optimizer.')


#########################
#   Default Settings    #
#########################
tf.app.flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')

tf.app.flags.DEFINE_boolean('grad_cam', False,
                            'grad_cam or cam.')

tf.app.flags.DEFINE_boolean('mean', True,
                            'grad_cam or cam.')

tf.app.flags.DEFINE_integer('worker_replicas', 1,
                            'Number of worker replicas.')

tf.app.flags.DEFINE_integer('num_ps_tasks', 0,
                            'The number of parameter servers. If the value is 0, then the parameters '
                            'are handled locally by the worker.')

tf.app.flags.DEFINE_integer('task', 0,
                            'Task id of the replica running the training.')

tf.app.flags.DEFINE_float('moving_average_decay', 0.9999,
                          'The decay to use for the moving average.'
                          'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer('input_queue_memory_factor', 32,
                            """Size of the queue of preprocessed images. """)

tf.app.flags.DEFINE_integer('num_readers', 4,
                            'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer('num_preprocessing_threads', 4,
                            'The number of threads used to create the batches.')

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

tf.app.flags.DEFINE_integer("min_after_dequeue", 128, 
                            "min nums data filename in queue")

tf.app.flags.DEFINE_integer("capacity", 200, 
                            "capacity")


FLAGS = tf.app.flags.FLAGS


def main(_):
    # create folders
    mkdir_if_missing(FLAGS.checkpoint_dir)
    mkdir_if_missing(FLAGS.log_dir)
    mkdir_if_missing(FLAGS.saliency_map)
    # training
    train_models.train()


if __name__ == '__main__':
    tf.app.run()
