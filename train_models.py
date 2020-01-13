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

def focal_loss(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
    
    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)
    
    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return tf.reduce_sum(per_entry_cross_ent)

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

def _tower_loss(network_fn, images, labels, cross = True, reuse=False, is_training=False):
    """Calculate the total loss on a single tower running the reid model.""" 
    net_logits, flat_logits, net_endpoints, net_raw_loss, net_pred, net_features, dice_loss, cross_loss = {}, {}, {}, {}, {}, {}, {}, {}
    weight_loss, offset = {},{}
    print(images)
    for i in range(FLAGS.num_networks):
        #tf.summary.image("label" , get_image_summary(tf.expand_dims(tf.argmax(labels,-1),-1)))
        net_logits["{0}".format(i)],net_endpoints["{0}".format(i)] = network_fn["{0}".format(i)](images, labels, reuse=reuse, is_training=is_training, scope=('dmlnet_%d' % i))
        net_pred["{0}".format(i)] = net_endpoints["{0}".format(i)]['Predictions']
        #offset["{0}".format(i)] = net_endpoints["{0}".format(i)]['offset2']
        offset["{0}".format(i)] = net_endpoints["{0}".format(i)]['Predictions']

        #### dice loss && cross entropy loss
        flat_logits["{0}".format(i)] = tf.reshape(net_logits["{0}".format(i)], [-1, FLAGS.num_classes])
        flat_labels = tf.reshape(labels, [-1, FLAGS.num_classes])
        if cross == True:
            eps = 1e-8
            prediction = pixel_wise_softmax(net_logits["{0}".format(i)])
            intersection = tf.reduce_sum(prediction * labels)
            union = eps + tf.reduce_sum(prediction) + tf.reduce_sum(labels)
            dice_loss["{0}".format(i)] = 1.-(intersection / (union-intersection))
            #cross_loss["{0}".format(i)] = tf.losses.softmax_cross_entropy(onehot_labels = flat_labels, logits = flat_logits["{0}".format(i)])
            cross_loss["{0}".format(i)] = focal_loss(flat_logits["{0}".format(i)], flat_labels, weights=None, alpha=0.5, gamma=3)
            net_raw_loss["{0}".format(i)] = dice_loss["{0}".format(i)] + cross_loss["{0}".format(i)]                
            kl_weight = 1.0
        else:
            dice_loss["{0}".format(i)] = tf.constant(0.0)
            #cross_loss["{0}".format(i)] = tf.losses.softmax_cross_entropy(onehot_labels = flat_labels, logits = flat_logits["{0}".format(i)])
            cross_loss["{0}".format(i)] = tf.constant(0.0)
            net_raw_loss["{0}".format(i)] = tf.constant(0.0)
            kl_weight = 1.0
        if i == 0:
            labels = 1.0-labels
            
    # Add KL loss if there are more than one network
    net_loss, kl_loss, overlap_loss, net_reg_loss, net_total_loss, net_loss_averages, net_loss_averages_op = {}, {}, {}, {}, {}, {}, {}
    kl, exclusion = {},{}

    for i in range(FLAGS.num_networks):
        net_loss["{0}".format(i)] = net_raw_loss["{0}".format(i)]
        for j in range(FLAGS.num_networks):
            if i != j:
                #### JS divergency loss && exclusion loss
                pred1 = tf.nn.softmax(flat_logits["{0}".format(i)])
                pred2 = 1.0-tf.nn.softmax(flat_logits["{0}".format(j)])    
                kl_loss["{0}{0}".format(i, j)] = kl_loss_compute(pred1, pred2)
                eps = 1e-8
                prediction1 = pixel_wise_softmax(net_logits["{0}".format(i)])
                prediction2 = pixel_wise_softmax(net_logits["{0}".format(j)])
                intersection = tf.reduce_sum(prediction1 * prediction2)
                union = eps + tf.reduce_sum(prediction1) + tf.reduce_sum(prediction2)
                overlap_loss["{0}{0}".format(i, j)] = (2 * intersection / (union))
                
                net_loss["{0}".format(i)] += kl_weight*(kl_loss["{0}{0}".format(i, j)]+overlap_loss["{0}{0}".format(i, j)])
                #tf.summary.scalar('kl_loss_%d%d' % (i, j), kl_loss["{0}{0}".format(i, j)])
                #tf.summary.scalar('overlap_loss_%d%d' % (i, j), overlap_loss["{0}{0}".format(i, j)])

        #kl["{0}".format(i)] = kl_loss["{0}{0}".format(i, 0)]+kl_loss["{0}{0}".format(i, 1)]
        #exclusion["{0}".format(i)] = overlap_loss["{0}{0}".format(i, 0)]+overlap_loss["{0}{0}".format(i, 1)]
        kl["{0}".format(i)] = tf.constant(0)
        exclusion["{0}".format(i)] = tf.constant(0)


        net_reg_loss["{0}".format(i)] = tf.add_n([FLAGS.weight_decay * tf.nn.l2_loss(var) for var in tf.trainable_variables() if 'dmlnet_%d' % i in var.name])
        #net_reg_loss["{0}".format(i)] = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=('dmlnet_%d' % i))
        #net_total_loss["{0}".format(i)] = tf.add_n([net_loss["{0}".format(i)]] +
        #                                           net_reg_loss["{0}".format(i)],
        #                                           name=('net%d_total_loss' % i))
        net_total_loss["{0}".format(i)] = net_loss["{0}".format(i)] + net_reg_loss["{0}".format(i)] 
        
        #tf.summary.scalar('net%d_loss_dice' % i, dice_loss["{0}".format(i)])
        #tf.summary.scalar('net%d_loss_cross' % i, cross_loss["{0}".format(i)])
        #tf.summary.scalar('net%d_loss_sum' % i, net_total_loss["{0}".format(i)])
        
        
    return net_total_loss, dice_loss, cross_loss, kl, exclusion, net_pred, offset

def make_png(iimage):
    oout = np.stack([iimage,iimage,iimage]).transpose([1,2,3,0])
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
        train_image_batch, train_label_batch = utils.get_image_label_batch(FLAGS, shuffle=True, name='test2')
        semi_image_batch, semi_label_batch = utils.get_image_label_batch(FLAGS, shuffle=True, name='train1_train2')
        test_image_batch, test_label_batch = utils.get_image_label_batch(FLAGS, shuffle=False, name='test1')
        train_x = train_image_batch[:,:,:,0:3]
        train_y = tf.cast((tf.squeeze(train_image_batch[:,:,:,3])+1.0)*0.5, tf.int32)
        train_y = tf.one_hot(train_y, depth = 2, axis=-1)
                            
        semi_x = semi_image_batch[:,:,:,0:3]
        semi_y = tf.cast((tf.squeeze(semi_image_batch[:,:,:,3])+1.0)*0.5, tf.int32)
        semi_y = tf.one_hot(semi_y, depth = 2, axis=-1) 
                            
        test_x = test_image_batch[:,:,:,0:3]
        test_y = tf.cast((tf.squeeze(test_image_batch[:,:,:,3])+1.0)*0.5, tf.int32)
        test_y = tf.one_hot(test_y, depth = 2, axis=-1) 

            
        precision, test_precision, val_precision, net_var_list, net_grads, net_update_ops, predictions, test_predictions,  val_predictions = {}, {}, {}, {}, {}, {}, {}, {}, {}
        semi_net_grads = {}

        with tf.name_scope('tower') as scope:
            with tf.variable_scope(tf.get_variable_scope()):
                net_loss, dice_loss, cross_loss, kl, exclusion, net_pred, offset = _tower_loss(network_fn, train_x, train_y, cross=True, reuse=False, is_training=True)
                semi_net_loss, _,_,_,_,_,_ = _tower_loss(network_fn, train_x, train_y, cross=False, reuse=True, is_training=True)
                test_net_loss, _,_,_,_,test_net_pred,test_offset=_tower_loss(network_fn,test_x,test_y, cross=True,reuse=True, is_training=False)

                truth = tf.argmax(train_y, axis=-1)
                test_truth = tf.argmax(test_y, axis=-1)

                # Reuse variables for the next tower.
                #tf.get_variable_scope().reuse_variables()

                # Retain the summaries from the final tower.
                #summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                var_list = tf.trainable_variables()

                for i in range(FLAGS.num_networks):
                    predictions["{0}".format(i)] = tf.argmax(net_pred["{0}".format(i)], axis=-1)
                    test_predictions["{0}".format(i)] = tf.argmax(test_net_pred["{0}".format(i)], axis=-1)
                #precision["{0}".format(0)] = tf.reduce_mean(tf.to_float(tf.equal(predictions["{0}".format(0)], truth)))
                #test_precision["{0}".format(0)] = tf.reduce_mean(tf.to_float(tf.equal(test_predictions["{0}".format(0)], test_truth)))

                precision["{0}".format(0)] = 2*tf.reduce_sum(tf.to_float(predictions["{0}".format(0)]) * tf.to_float(truth))/tf.reduce_sum(tf.to_float(predictions["{0}".format(0)] + truth))
                test_precision["{0}".format(0)] = 2*tf.reduce_sum(tf.to_float(test_predictions["{0}".format(0)]) * tf.to_float(test_truth))/tf.reduce_sum(tf.to_float(test_predictions["{0}".format(0)] + test_truth))
                
                #precision["{0}".format(1)] = tf.reduce_mean(tf.to_float(tf.equal(predictions["{0}".format(1)], 1-truth)))
               # test_precision["{0}".format(1)] = tf.reduce_mean(tf.to_float(tf.equal(test_predictions["{0}".format(1)], 1-test_truth)))

                #precision["{0}".format(1)] = 2*tf.reduce_sum(tf.to_float(predictions["{0}".format(1)]) * tf.to_float(1-truth))/tf.reduce_sum(tf.to_float(predictions["{0}".format(1)] + 1-truth))
                #test_precision["{0}".format(1)] = 2*tf.reduce_sum(tf.to_float(test_predictions["{0}".format(1)]) * tf.to_float(1-test_truth))/tf.reduce_sum(tf.to_float(test_predictions["{0}".format(1)] +1- test_truth))
   
                Gamma ={}
                for i in range(FLAGS.num_networks):
                    # Add a summary to track the training precision.
                    #summaries.append(tf.summary.scalar('precision_%d' % i, precision["{0}".format(i)]))
                    #summaries.append(tf.summary.scalar('test_precision_%d' % i, test_precision["{0}".format(i)]))
                    #summaries.append(tf.summary.scalar('val_precision_%d' % i, test_precision["{0}".format(i)]))

                    net_update_ops["{0}".format(i)] = \
                                tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=('%sdmlnet_%d' % (scope, i)))

                    net_var_list["{0}".format(i)] = \
                                    [var for var in var_list if 'dmlnet_%d' % i in var.name]

                    net_grads["{0}".format(i)] = net_opt["{0}".format(i)].compute_gradients(
                                    net_loss["{0}".format(i)], var_list=net_var_list["{0}".format(i)])
                
                    semi_net_grads["{0}".format(i)] = net_opt["{0}".format(i)].compute_gradients(
                                    semi_net_loss["{0}".format(i)], var_list=net_var_list["{0}".format(i)])
                Gamma["{0}".format(0)], Gamma["{0}".format(1)] = {},{} 
                for var in tf.trainable_variables():
                    if 'dmlnet_0' in var.name and 'GGamma' in var.name:
                        Gamma["{0}".format(0)][var.name] = var
                    if 'dmlnet_1' in var.name and 'GGamma' in var.name:
                        Gamma["{0}".format(1)][var.name] = var

        #################################
        # Configure the moving averages #
        #################################

        if FLAGS.moving_average_decay:
            moving_average_variables = {}
            all_moving_average_variables = slim.get_model_variables()
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, global_step)
            for i in range(FLAGS.num_networks):
                moving_average_variables["{0}".format(i)] = \
                    [var for var in all_moving_average_variables if 'dmlnet_%d' % i in var.name]
                net_update_ops["{0}".format(i)].append(
                    variable_averages.apply(moving_average_variables["{0}".format(i)]))

        # Apply the gradients to adjust the shared variables.
        net_grad_updates, net_train_op, semi_net_grad_updates, semi_net_train_op = {}, {}, {}, {}
        for i in range(FLAGS.num_networks):
            net_grad_updates["{0}".format(i)] = net_opt["{0}".format(i)].apply_gradients(
                net_grads["{0}".format(i)], global_step=global_step)
            #semi_net_grad_updates["{0}".format(i)] = semi_net_opt["{0}".format(i)].apply_gradients(
            #    semi_net_grads["{0}".format(i)], global_step=global_step)
            net_update_ops["{0}".format(i)].append(net_grad_updates["{0}".format(i)])
            #net_update_ops["{0}".format(i)].append(semi_net_grad_updates["{0}".format(i)])
            # Group all updates to into a single train op.
            net_train_op["{0}".format(i)] = tf.group(*net_update_ops["{0}".format(i)])
            
        '''# Apply the gradients to adjust the shared variables.
        net_train_op, semi_net_train_op = {}, {}
        for i in range(FLAGS.num_networks):
            net_train_op["{0}".format(i)] = net_opt["{0}".format(i)].minimize(net_loss["{0}".format(i)], global_step=global_step, var_list=net_var_list["{0}".format(i)])
            #semi_net_train_op["{0}".format(i)] = semi_net_opt["{0}".format(i)].minimize(semi_net_loss["{0}".format(i)],global_step=global_step, var_list=net_var_list["{0}".format(i)])'''

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
        
        #load_fn = slim.assign_from_checkpoint_fn(os.path.join(FLAGS.checkpoint_dir, 'model.ckpt-10'),tf.global_variables(),ignore_missing_vars=True)
        #load_fn = slim.assign_from_checkpoint_fn('./WCE_densenet4/checkpoint/model.ckpt-70',tf.global_variables(),ignore_missing_vars=False)
        #load_fn(sess)
        
        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        #summary_writer = tf.summary.FileWriter(
        #    os.path.join(FLAGS.log_dir),
        #    graph=sess.graph)

        net_loss_value, test_net_loss_value, precision_value, test_precision_value = {}, {}, {}, {}
        dice_loss_value, cross_loss_value, kl_value, exclusion_value = {}, {},{}, {}
        parameters = utils.count_trainable_params()
        print("Total training params: %.1fM \r\n" % (parameters / 1e6))
                  
        start_time = time.time() 
        counter = 0
        infile = open(os.path.join(FLAGS.log_dir, 'log.txt'),'w')
        batch_count = np.int32(15552 / FLAGS.batch_size)
        
        GGamma,GG={},{}
        GGamma["{0}".format(0)],GGamma["{0}".format(1)] = {},{}
        for i in range(FLAGS.num_networks):
            _, GG["{0}".format(i)] = sess.run([precision["{0}".format(i)],Gamma["{0}".format(i)]])
            for k in GG["{0}".format(i)].keys():
                if 'dmlnet_%d' % i in k and 'GGamma' in k:
                    GGamma_key = k.split(':')[0]
                    GGamma_key = '_'.join(GGamma_key.split('/'))
                    GGamma["{0}".format(i)][GGamma_key]=[float(GG["{0}".format(i)][k])]
                                          
                
        for epoch in range(1, 1+FLAGS.max_number_of_epochs):
            if (epoch) % 40 == 0: 
                FLAGS.learning_rate = FLAGS.learning_rate * 0.1
            #if (epoch) % 75 == 0: 
            #    FLAGS.learning_rate = FLAGS.learning_rate * 0.1    
                
            for batch_idx in range(batch_count):
                counter += 1
                for i in range(FLAGS.num_networks):
                    _, net_loss_value["{0}".format(i)], dice_loss_value["{0}".format(i)], cross_loss_value["{0}".format(i)],kl_value["{0}".format(i)], exclusion_value["{0}".format(i)], precision_value["{0}".format(i)], offset_map = \
                    sess.run([net_train_op["{0}".format(i)], net_loss["{0}".format(i)],dice_loss["{0}".format(i)],cross_loss["{0}".format(i)], kl["{0}".format(i)], exclusion["{0}".format(i)], precision["{0}".format(i)], offset["{0}".format(i)]])
                    assert not np.isnan(net_loss_value["{0}".format(i)]), 'Model diverged with loss = NaN'
                    #if epoch >= 20:
                    #    _ = sess.run([sc_update_op["{0}".format(i)]])
                    
                if batch_idx % 500 == 0:
                    for i in range(FLAGS.num_networks):
                        test_net_loss_value["{0}".format(i)],GG["{0}".format(i)], test_precision_value["{0}".format(i)] = sess.run([test_net_loss["{0}".format(i)], Gamma["{0}".format(i)], test_precision["{0}".format(i)]])
                        
                        for k in GG["{0}".format(i)].keys():
                            if 'dmlnet_%d' % i in k and 'GGamma' in k:
                                GGamma_key = k.split(':')[0]
                                GGamma_key = '_'.join(GGamma_key.split('/'))
                                GGamma["{0}".format(i)][GGamma_key].extend([float(GG["{0}".format(i)][k])])
                                          
                    #format_str = 'Epoch: [%3d] [%3d/%3d] net0loss = %.4f, net0acc = %.4f, net0testloss = %.4f, net0testacc = %.4f,   net1loss = %.4f, net1acc = %.4f, net1testloss = %.4f, net1testacc = %.4f'
                    #print(format_str % (epoch, batch_idx,batch_count, net_loss_value["{0}".format(0)],
                    #      precision_value["{0}".format(0)],test_net_loss_value["{0}".format(0)],np.float32(test_precision_value["{0}".format(0)]),net_loss_value["{0}".format(1)],precision_value["{0}".format(1)],test_net_loss_value["{0}".format(1)],np.float32(test_precision_value["{0}".format(1)])))
                    #format_str1 = 'Epoch: [%3d] [%3d/%3d] time: %4.3f, dice0 = %.5f, cross0 = %.4f, kl0 = %.4f, exclusion0 = %.4f,     dice1 = %.5f, cross1 = %.4f, kl1 = %.4f, exclusion1 = %.4f'
                    #print(format_str1 % (epoch, batch_idx,batch_count, time.time()-start_time, dice_loss_value["{0}".format(0)],cross_loss_value["{0}".format(0)],np.float32(kl_value["{0}".format(0)]),np.float32(exclusion_value["{0}".format(0)]),dice_loss_value["{0}".format(1)],cross_loss_value["{0}".format(1)],np.float32(kl_value["{0}".format(1)]),np.float32(exclusion_value["{0}".format(1)])))
                    
                    print(offset_map.max())
                    '''infile.write(format_str % (epoch, batch_idx,batch_count, net_loss_value["{0}".format(0)], precision_value["{0}".format(0)],test_net_loss_value["{0}".format(0)],np.float32(test_precision_value["{0}".format(0)]),net_loss_value["{0}".format(1)],precision_value["{0}".format(1)],test_net_loss_value["{0}".format(1)],np.float32(test_precision_value["{0}".format(1)])))
                    infile.write('\n')
                    infile.write(format_str1 % (epoch, batch_idx,batch_count, time.time()-start_time, dice_loss_value["{0}".format(0)],cross_loss_value["{0}".format(0)],np.float32(kl_value["{0}".format(0)]),np.float32(exclusion_value["{0}".format(0)]),dice_loss_value["{0}".format(1)],cross_loss_value["{0}".format(1)],np.float32(kl_value["{0}".format(1)]),np.float32(exclusion_value["{0}".format(1)])))
                    infile.write('\n')'''
                    format_str = 'Epoch: [%3d] [%3d/%3d] time: %4.4f, net0_loss = %.5f, net0_acc = %.4f, net0_test_acc = %.4f'
                    print(format_str % (epoch, batch_idx,batch_count, time.time()-start_time, net_loss_value["{0}".format(0)],
                           precision_value["{0}".format(0)],np.float32(test_precision_value["{0}".format(0)])))
                    
                    #format_str = 'Epoch: [%3d] [%3d/%3d] time: %4.4f, net0_loss = %.5f, net0_acc = %.4f'
                    #print(format_str % (epoch, batch_idx,batch_count, time.time()-start_time, net_loss_value["{0}".format(1)],
                    #     precision_value["{0}".format(1)]))

                if batch_idx == 0:
                    testpred0, test_gt,test_X = sess.run([test_predictions["{0}".format(0)], test_truth, test_x])
                    tot_num_samples = FLAGS.batch_size
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    save_images(make_png(test_gt[:manifold_h * manifold_w, :,:]),[manifold_h, manifold_w], os.path.join(FLAGS.saliency_map, str(epoch)+'test_gt.jpg'))
                    save_images(test_X[:manifold_h * manifold_w, :,:,:],[manifold_h, manifold_w], os.path.join(FLAGS.saliency_map, str(epoch)+'test.jpg'))
                    save_images(make_png(testpred0[:manifold_h * manifold_w, :,:]),[manifold_h, manifold_w],os.path.join(FLAGS.saliency_map, str(epoch)+'test_pred0.jpg'))     
                    #save_images(make_png(testpred1[:manifold_h * manifold_w, :,:]),[manifold_h, manifold_w],os.path.join(FLAGS.saliency_map, str(epoch)+'test_pred1.jpg')) 

                    #for index in range(test_gt.shape[0]):
                    #    scipy.misc.imsave(os.path.join(FLAGS.saliency_map, str(epoch)+'_'+str(index)+'test.jpg'), make_png(test_gt[index,:,:]))
                    #    scipy.misc.imsave(os.path.join(FLAGS.saliency_map, str(epoch)+'_'+str(index)+'test_pred0.jpg'), make_png(testpred["{0}".format(0)][index,:,:]))
                    #    scipy.misc.imsave(os.path.join(FLAGS.saliency_map, str(epoch)+'_'+str(index)+'test_pred1.jpg'), make_png(testpred["{0}".format(1)][index,:,:]))
                
                #summary_str = sess.run(summary_op)
                #summary_writer.add_summary(summary_str, counter)
                
            # Save the model checkpoint periodically.
            if epoch % FLAGS.ckpt_steps == 0 or epoch == FLAGS.max_number_of_epochs:
                checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=epoch)
                
        for i in range(FLAGS.num_networks):
            for k in GG["{0}".format(i)].keys():
                if 'dmlnet_%d' % i in k and 'GGamma' in k:
                    GGamma_key = k.split(':')[0]
                    GGamma_key = '_'.join(GGamma_key.split('/'))
                    gamma_file = open(os.path.join(FLAGS.log_dir, GGamma_key+'.txt'), 'w')
                    for g in GGamma["{0}".format(i)][GGamma_key]:
                        gamma_file.write(str(g)+' \n')
                    gamma_file.close()

