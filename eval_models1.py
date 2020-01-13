"""
    Generic training script that trains a model using a given dataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory
from datasets.utils import *
import numpy as np
import time
import utils
from ops import *
from utils import *
import utils
from sklearn.metrics import *
import cv2
slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS
from tensorflow.python.ops import array_ops
from nets.tf_unet.layers import (weight_variable, weight_variable_devonc, bias_variable,
                            conv2d, deconv2d, max_pool, crop_and_concat, pixel_wise_softmax,
                            cross_entropy)
def _average_gradients(tower_grads, catname=None):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(input=g, axis=0)
            # print(g)
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads, name=catname)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def kl_loss_compute(pred1, pred2):
    """ JS loss
    """    
    ave = (pred1 + pred2) / 2
    loss = 0.5*tf.reduce_mean(tf.reduce_sum(pred2 * tf.log((1e-8 + pred2) / (ave + 1e-8)), 1)) + 0.5*tf.reduce_mean(tf.reduce_sum(pred1 * tf.log((1e-8 + pred1) / (ave + 1e-8)), 1))
    #loss = tf.reduce_mean(tf.reduce_sum(pred2 * tf.log((1e-8 + pred2) / (pred1 + 1e-8)), 1))             
    #loss = 0.5*tf.reduce_mean(tf.reduce_sum(pred2 * tf.log((1e-8 + pred2) / (pred1 + 1e-8)), 1)) + 0.5*tf.reduce_mean(tf.reduce_sum(pred1 * tf.log((1e-8 + pred1) / (pred2 + 1e-8)), 1))
    return loss

def rank_loss(logits1, logits2, labels):
    margin = 0.1
    softmax1 = tf.nn.softmax(logits1,axis=-1)
    softmax2 = tf.nn.softmax(logits2,axis=-1)
    label_prob1 = tf.reduce_sum(tf.multiply(softmax1,labels),axis=-1)
    label_prob2 = tf.reduce_sum(tf.multiply(softmax2,labels),axis=-1)
    loss = tf.nn.relu(label_prob1-label_prob2+margin)
    return loss   
    
def get_center_loss(Features, Labels, alpha, num_classes, scope, reuse):
    with tf.variable_scope(scope, reuse=reuse):
        len_features = Features.get_shape()[1]
        centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
            initializer=tf.constant_initializer(0), trainable=False)

        Labels = tf.reshape(Labels, [-1])
    
        centers_batch = tf.gather(centers, Labels)
        numerator = tf.norm(Features - centers_batch, axis=-1)
        f = tf.expand_dims(Features, axis=1)
        f = tf.tile(f,[1,centers.shape[0],1])
        denominator = tf.norm(f - centers, axis=-1)
        denominator = 1e-8 + tf.reduce_sum(denominator, axis=-1) - numerator
        loss_weight = (num_classes-1) * numerator/denominator
    
        diff = centers_batch - Features
    
        unique_label, unique_idx, unique_count = tf.unique_with_counts(Labels)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])
    
        diff = diff / tf.cast((1 + appear_times), tf.float32)
        diff = alpha * diff    
        centers_update_op = tf.scatter_sub(centers, Labels, diff)
        
    return loss_weight, centers, centers_update_op

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

def _tower_loss(network_fn, images, labels, is_cross = True, reuse=False, is_training=False):
    """Calculate the total loss on a single tower running the reid model.""" 
    net_logits, flat_logits, net_endpoints, net_raw_loss, net_pred, net_features, dice_loss, cross_loss = {}, {}, {}, {}, {}, {}, {}, {}
    offset3, offset2 = {},{}
    print(images)
    for i in range(FLAGS.num_networks):
        #tf.summary.image("label" , get_image_summary(tf.expand_dims(tf.argmax(labels,-1),-1)))
        net_logits["{0}".format(i)],net_endpoints["{0}".format(i)] = network_fn["{0}".format(i)](images, labels, reuse=reuse, is_training=is_training, scope=('dmlnet_%d' % i))
        net_pred["{0}".format(i)] = net_endpoints["{0}".format(i)]['Predictions']

    return net_pred
def make_png1(i1image, gt):
    iimage = np.copy(i1image)
    iimage = cv2.resize(iimage,(128,128))
    iimage = 255.0-iimage*100.0
    oout = np.int32(np.stack([iimage,iimage,iimage]).transpose([1,2,0]))

    return oout
def make_png(iimage):
    iimage = cv2.resize(iimage,(128,128))
    oout = np.stack([iimage,iimage,iimage]).transpose([1,2,0])
    return oout

    
def train():
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        #######################
        # Config model_deploy #
        #######################
        deploy_config = model_deploy.DeploymentConfig(
            num_clones=FLAGS.num_clones,
            clone_on_cpu=FLAGS.clone_on_cpu,
            replica_id=FLAGS.task,
            num_replicas=FLAGS.worker_replicas,
            num_ps_tasks=FLAGS.num_ps_tasks)

        # Create global_step
        with tf.device(deploy_config.variables_device()):
            global_step = tf.train.create_global_step()

        ######################
        # Select the network and #
        ######################
        network_fn = {}
        model_names = [net.strip() for net in FLAGS.model_name.split(',')]
        for i in range(FLAGS.num_networks):
            network_fn["{0}".format(i)] = nets_factory.get_network_fn(
                model_names[i],
                num_classes=FLAGS.num_classes,
                weight_decay=FLAGS.weight_decay)
            
        #########################################
        # Configure the optimization procedure. #
        #########################################
        with tf.device(deploy_config.optimizer_device()):
            net_opt, semi_net_opt = {}, {}
            for i in range(FLAGS.num_networks):
                net_opt["{0}".format(i)] = tf.train.AdamOptimizer(FLAGS.learning_rate,
                                                                  beta1=FLAGS.adam_beta1,
                                                                  beta2=FLAGS.adam_beta2,
                                                                  epsilon=FLAGS.opt_epsilon)
                semi_net_opt["{0}".format(i)] = tf.train.AdamOptimizer(FLAGS.learning_rate,
                                                                  beta1=FLAGS.adam_beta1,
                                                                  beta2=FLAGS.adam_beta2,
                                                                  epsilon=FLAGS.opt_epsilon)

        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name  # or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=True)

        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################            
        train_image_batch, train_label_batch = utils.get_image_label_batch(FLAGS, shuffle=True, name='train4')
        test_image_batch, test_label_batch = utils.get_image_label_batch(FLAGS, shuffle=False, name='test4')
        train_x = train_image_batch[:,:,:,0:3]
        train_y = tf.cast((tf.squeeze(train_image_batch[:,:,:,3])+1.0)*0.5, tf.int32)
        train_y = tf.one_hot(train_y, depth = 2, axis=-1)
                            
        test_x = test_image_batch[:,:,:,0:3]
        test_y = tf.cast((tf.squeeze(test_image_batch[:,:,:,3])+1.0)*0.5, tf.int32)
        test_y = tf.one_hot(test_y, depth = 2, axis=-1) 
            
        precision, test_precision, val_precision, net_var_list, net_grads, net_update_ops, predictions, test_predictions,  val_predictions = {}, {}, {}, {}, {}, {}, {}, {}, {}
        #semi_net_grads = {}

        with tf.name_scope('tower') as scope:
            with tf.variable_scope(tf.get_variable_scope()):
                net_pred = _tower_loss(network_fn, train_x, train_y, is_cross = True, reuse=False, is_training=False)
                #semi_net_loss, sc_update_op, semi_net_pred, semi_attention0, semi_attention1, semi_second_input = _tower_loss(network_fn, semi_image_batch, semi_label_batch, is_cross = False, reuse=True, is_training=True)
                test_net_pred  = _tower_loss(network_fn, test_x, test_y, is_cross = True, reuse=True, is_training=False)

                truth = tf.argmax(train_y, axis=-1)
                test_truth = tf.argmax(test_y, axis=-1)

                # Reuse variables for the next tower.
                #tf.get_variable_scope().reuse_variables()

                # Retain the summaries from the final tower.
                summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                var_list = tf.trainable_variables()

                for i in range(FLAGS.num_networks):
                    predictions["{0}".format(i)] = tf.argmax(net_pred["{0}".format(i)], axis=-1)
                test_predictions["{0}".format(0)] = tf.argmax(test_net_pred["{0}".format(0)], axis=-1)
                
                precision["{0}".format(0)] = tf.reduce_mean(tf.to_float(tf.equal(predictions["{0}".format(0)], truth)))
                test_precision["{0}".format(0)] = tf.reduce_mean(tf.to_float(tf.equal(test_predictions["{0}".format(0)], test_truth)))
                
                    # Add a summary to track the training precision.
                    #summaries.append(tf.summary.scalar('precision_%d' % i, precision["{0}".format(i)]))
                    #summaries.append(tf.summary.scalar('test_precision_%d' % i, test_precision["{0}".format(i)]))
          
        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation from the last tower summaries.
        #summary_op = tf.summary.merge(summaries)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85),
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)
        
        load_fn = slim.assign_from_checkpoint_fn(os.path.join(FLAGS.checkpoint_dir, 'model.ckpt-50'),tf.global_variables(),ignore_missing_vars=True)
        #load_fn = slim.assign_from_checkpoint_fn('./WCE_densenet4/checkpoint/model.ckpt-20',tf.global_variables(),ignore_missing_vars=True)
        load_fn(sess)
        
        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        #summary_writer = tf.summary.FileWriter(
        #    os.path.join(FLAGS.log_dir),
        #    graph=sess.graph)

        net_loss_value, test_precision_value, test_predictions_value, precision_value = {}, {}, {}, {}

        parameters = utils.count_trainable_params()
        print("Total training params: %.1fM \r\n" % (parameters / 1e6))
                  
        start_time = time.time() 
        counter = 0
        infile = open(os.path.join(FLAGS.log_dir, 'result.txt'),'w')
        batch_count = np.int32(5184 / FLAGS.batch_size)  
        testpred, testprecision, valpred, valprecision = {},{},{},{}
        test_DI, test_JA, val_DI, val_JA = {},{},{},{} 
        acc0 = 0
        acc1 = 0
        sen0 = 0
        sen1 = 0
        spe0 = 0
        spe1 = 0
        di0 = 0
        di1 = 0
        ja0 = 0
        ja1 = 0
        for batch_idx in range(batch_count):
            #for i in range(FLAGS.num_networks):            
            testpred, test_gt, Test_x = sess.run([test_predictions, test_truth, test_x])
            for index in range(test_truth.shape[0]):
                scipy.misc.imsave(os.path.join(FLAGS.saliency_map, str(batch_idx)+'_'+str(index)+'test.jpg'), Test_x[index,:,:,:])
                scipy.misc.imsave(os.path.join(FLAGS.saliency_map, str(batch_idx)+'_'+str(index)+'test_gt.jpg'), make_png(test_gt[index,:,:]))
                scipy.misc.imsave(os.path.join(FLAGS.saliency_map, str(batch_idx)+'_'+str(index)+'test_pred0.jpg'), make_png(testpred["{0}".format(0)][index,:,:]))
            
            testpred0 = np.int32(np.reshape(testpred["{0}".format(0)],[-1]))
            test_gt0 = np.int32(np.reshape(test_gt,[-1]))
            #print(testpred0.shape, testpred1.shape, test_gt0.shape, test_gt1.shape)
            
            tn0, fp0, fn0, tp0 = confusion_matrix(test_gt0,testpred0).ravel()

            net0_acc = (tp0 + tn0)/(tp0 + fp0 + tn0 +fn0)
            net0_sen = tp0/(tp0+fn0)
            net0_spe = tn0/(tn0+fp0)
            net0_DI = 2*tp0/(2*tp0+fn0+fp0)
            net0_JA = tp0/(tp0+fn0+fp0)
            acc0 += net0_acc
            sen0 += net0_sen
            spe0 += net0_spe
            di0 += net0_DI
            ja0 += net0_JA
            
            print('test  :acc     dice     jac     sen     spe')
            format_str = 'test_net0: %.6f     %.6f     %.6f     %.6f     %.6f'
            print(format_str % (net0_acc, net0_DI, net0_JA, net0_sen, net0_spe))
            infile.write('test  :acc     dice     jac     sen     spe\n')
            infile.write(format_str % (net0_acc, net0_DI, net0_JA, net0_sen, net0_spe) + ' \n')

        acc0 = acc0/batch_count
        sen0 = sen0/batch_count
        spe0 = spe0/batch_count
        di0 = di0/batch_count
        ja0 = ja0/batch_count
        print('test_mean  :acc0     dice0     jac0     sen0     spe0 ')
        format_str1 = 'test: acc0: %.6f     %.6f     %.6f     %.6f     %.6f  '
        print(format_str1 % (acc0,  di0, ja0, sen0, spe0)) 
        infile.write('test_mean  :acc0     dice0     jac0     sen0     spe0 \n')
        infile.write(format_str1 % (acc0,  di0, ja0, sen0, spe0) + ' \n\n\n')
        
        
        infile.close()
