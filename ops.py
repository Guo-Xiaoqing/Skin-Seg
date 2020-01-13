import tensorflow as tf
import tensorflow.contrib as tf_contrib
import numpy as np

# Xavier : tf_contrib.layers.xavier_initializer()
# He : tf_contrib.layers.variance_scaling_initializer()
# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# l2_decay : tf_contrib.layers.l2_regularizer(0.0001)

weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
weight_regularizer = None

##################################################################################
# Layer
##################################################################################

def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', scope='conv_0'):
    with tf.variable_scope(scope):
        if pad_type == 'zero' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        if pad_type == 'reflect' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')

        #x = tf.layers.conv2d(inputs=x, filters=channels,
        #                     kernel_size=kernel, kernel_initializer=weight_init,
        #                     kernel_regularizer=weight_regularizer,
        #                     strides=stride, use_bias=use_bias)
        x = tf.contrib.layers.conv2d(inputs=x, num_outputs=channels, kernel_size=kernel, 
                                         stride=stride, padding='VALID',
                                         activation_fn=None,
                                         weights_initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
        x = tf.nn.bias_add(x, bias)

    return x

def atrous_conv2d(x, channels, kernel=3, rate=2, pad=0, pad_type='zero', scope='conv_0'):
    with tf.variable_scope(scope):
        if pad_type == 'zero' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        if pad_type == 'reflect' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')
             
        w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
        x = tf.nn.atrous_conv2d(value=x, filters=w, rate=2, padding='SAME')
        bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
        x = tf.nn.bias_add(x, bias)

    return x

def flatten(x) :
    return tf.layers.flatten(x)

def hw_flatten(x) :
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])
#########################
#adaptive dilated conv
#########################

# Definition of the regular 2D convolutional
def adaptive_conv(x, kernel_size, stride, output_channals, mode, reuse=True):
    with tf.variable_scope(mode, reuse=reuse):
        if mode == 'offset':
            layer_output = tf.layers.conv2d(x, filters=output_channals, kernel_size=kernel_size, strides=stride, padding='SAME', kernel_initializer = tf.zeros_initializer(), bias_initializer = tf.ones_initializer())
            layer_output = tf.clip_by_value(layer_output, -0.25*int(x.shape[1]), 0.25*int(x.shape[1]))
        if mode == 'weight':
            layer_output = tf.layers.conv2d(x, filters=output_channals, kernel_size=kernel_size, strides=stride, padding='SAME', bias_initializer = tf.zeros_initializer())
        if mode == 'feature':
            layer_output = tf.layers.conv2d(x, filters=output_channals, kernel_size=kernel_size, strides=kernel_size, padding='SAME', kernel_initializer = weight_init, bias_initializer = weight_init)   
        #layer_output = conv(x, output_channals, kernel=kernel_size, stride=kernel_size, sn=True, scope='feature')
    return layer_output

# Create the pn [1, 1, 1, 2N]
def adaptive_pn(kernel_size, dtype):
    pn_x, pn_y = np.meshgrid(range(-(kernel_size-1)//2, (kernel_size-1)//2+1), range(-(kernel_size-1)//2, (kernel_size-1)//2+1), indexing="ij")

    # The order is [x1, x2, ..., y1, y2, ...]
    pn = np.concatenate((pn_x.flatten(), pn_y.flatten()))

    #pn = np.reshape(pn, [1, 1, 1, 2 * kernel_size ** 2])

    # Change the dtype of pn
    pn = tf.constant(pn, dtype)

    return pn

# Create the p0 [1, h, w, 2N]
def adaptive_p0(kernel_size, x_size, dtype):

    bs, h, w, C = x_size

    p0_x, p0_y = np.meshgrid(range(0, h), range(0, w), indexing="ij")
    p0_x = p0_x.flatten().reshape(1, h, w, 1).repeat(kernel_size ** 2, axis=3)
    p0_y = p0_y.flatten().reshape(1, h, w, 1).repeat(kernel_size ** 2, axis=3)
    p0 = np.concatenate((p0_x, p0_y), axis=3)

    # Change the dtype of p0
    p0 = tf.constant(p0, dtype)

    return p0

def adaptive_q(x_size, dtype):

    bs, h, w, c = x_size

    q_x, q_y = np.meshgrid(range(0, h), range(0, w), indexing="ij")
    q_x = q_x.flatten().reshape(h, w, 1)
    q_y = q_y.flatten().reshape(h, w, 1)
    q = np.concatenate((q_x, q_y), axis=2)

    # Change the dtype of q
    q = tf.constant(q, dtype)

    return q

def adaptive_reshape_x_offset(x_offset, kernel_size):

    bs, h, w, N, C = x_offset.get_shape().as_list()

    # Get the new_shape
    new_shape = [bs, h, w * kernel_size, C]
    x_offset = [tf.reshape(x_offset[:, :, :, s:s+kernel_size, :], new_shape) for s in range(0, N, kernel_size)]
    x_offset = tf.concat(x_offset, axis=2)

    # Reshape to final shape [batch_size, h*kernel_size, w*kernel_size, C]
    x_offset = tf.reshape(x_offset, [bs, h * kernel_size, w * kernel_size, C])

    return x_offset

def adaptive_deform_con2v(input1, input2, num_outputs, kernel_size, stride, trainable, name, reuse):
    N = kernel_size ** 2 
    with tf.variable_scope(name, reuse=reuse):
        bs, h, w, C = input1.get_shape().as_list()
        
        # offset with shape [batch_size, h, w, 1]
        offset = adaptive_conv(input1, kernel_size, stride, 1, "offset", reuse=reuse)
        #print(offset)

        # delte_weight with shape [batch_size, h, w, N * C]
        #delte_weight = adaptive_conv(input, kernel_size, stride, N * C, "weight")
        #delte_weight = tf.sigmoid(delte_weight)

        # pn with shape [1, 1, 1, 2N]
        pn = adaptive_pn(kernel_size, offset.dtype)
        #print(pn)

        # p0 with shape [1, h, w, 2N]
        p0 = adaptive_p0(kernel_size, [bs, h, w, C], offset.dtype)
        #print(p0)

        # p with shape [batch_size, h, w, 2N]
        #p = pn + p0 + offset
        p = offset*pn + p0
        #print(p)

        # Reshape p to [batch_size, h, w, 2N, 1, 1]
        p = tf.reshape(p, [bs, h, w, 2 * N, 1, 1])

        # q with shape [h, w, 2]
        q = adaptive_q([bs, h, w, C], offset.dtype)

        # Bilinear interpolation kernel G ([batch_size, h, w, N, h, w])
        gx = tf.maximum(1 - tf.abs(p[:, :, :, :N, :, :] - q[:, :, 0]), 0)
        gy = tf.maximum(1 - tf.abs(p[:, :, :, N:, :, :] - q[:, :, 1]), 0)
        G = gx * gy

        # Reshape G to [batch_size, h*w*N, h*w]
        G = tf.reshape(G, [bs, h * w * N, h * w])

        # Reshape x to [batch_size, h*w, C]
        x = tf.reshape(input2, [bs, h*w, C])

        # x_offset with shape [batch_size, h, w, N, C]
        x = tf.reshape(tf.matmul(G, x), [bs, h, w, N, C])

        # Reshape x_offset to [batch_size, h*kernel_size, w*kernel_size, C]
        x = adaptive_reshape_x_offset(x, kernel_size)

        # Reshape delte_weight to [batch_size, h*kernel_size, w*kernel_size, C]
        #delte_weight = tf.reshape(delte_weight, [batch_size, h*kernel_size, w*kernel_size, C])

        #y = x_offset * delte_weight

        # Get the output of the deformable convolutional layer
        x = adaptive_conv(x, kernel_size, stride, num_outputs, "feature", reuse=reuse)

    return x, offset

'''def deform_conv2d(input1, input2, offset_kernel_size, kernel_size, num_outputs, activation=tf.nn.relu, scope="f", reuse=True):
    ''''''
    Args:
        x - 4D tensor [batch, i_h, i_w, i_c] NHWC format
        offset_shape - list with 4 elements
            [o_h, o_w, o_ic, o_oc]
        filter_shape - list with 4 elements
            [f_h, f_w, f_ic, f_oc]
            
            input1 pn = adaptive_pn(kernel_size, offset.dtype)= generate offset
            input2 = feature convolution
    ''''''

    offset_shape = [offset_kernel_size, offset_kernel_size, int(input1.shape[-1]), 1]
    filter_shape = [kernel_size, kernel_size, int(input2.shape[-1]), num_outputs]
    batch, i_h, i_w, i_c = input1.get_shape().as_list()
    f_h, f_w, f_ic, f_oc = filter_shape
    o_h, o_w, o_ic, o_oc = offset_shape
    assert f_ic==i_c and o_ic==i_c, "# of input_channel should match but %d, %d, %d"%(i_c, f_ic, o_ic)
    #assert o_oc==2*f_h*f_w, "# of output channel in offset_shape should be 2*filter_height*filter_width but %d and %d"%(o_oc, 2*f_h*f_w)

    with tf.variable_scope(scope or "deform_conv", reuse=reuse):
        offset = adaptive_conv(input1, kernel_size=offset_kernel_size, stride=1, output_channals=1, mode="offset", reuse=reuse)
        #offset = conv2d(input1, offset_shape, padding=True, scope="offset_conv") # offset : [batch, i_h, i_w, 1]
        pn = adaptive_pn(kernel_size, offset.dtype)# pn : [batch, i_h, i_w, o_oc(=2*f_h*f_w)]
    offset_map = offset*pn # offset_map : [batch, i_h, i_w, o_oc(=2*f_h*f_w)]
    offset_map = tf.reshape(offset_map, [batch, i_h, i_w, f_h, f_w, 2])
    offset_map_h = tf.tile(tf.reshape(offset_map[...,0], [batch, i_h, i_w, f_h, f_w]), [i_c,1,1,1,1]) # offset_map_h [batch*i_c, i_h, i_w, f_h, f_w]
    offset_map_w = tf.tile(tf.reshape(offset_map[...,1], [batch, i_h, i_w, f_h, f_w]), [i_c,1,1,1,1]) # offset_map_w [batch*i_c, i_h, i_w, f_h, f_w]

    coord_w, coord_h = tf.meshgrid(tf.range(i_w, dtype=tf.float32), tf.range(i_h, dtype=tf.float32)) # coord_w : [i_h, i_w], coord_h : [i_h, i_w]
    coord_fw, coord_fh = tf.meshgrid(tf.range(f_w, dtype=tf.float32), tf.range(f_h, dtype=tf.float32)) # coord_fw : [f_h, f_w], coord_fh : [f_h, f_w]
    ''''''
    coord_w 
        [[0,1,2,...,i_w-1],...]
    coord_h
        [[0,...,0],...,[i_h-1,...,i_h-1]]
    ''''''
    coord_h = tf.tile(tf.reshape(coord_h, [1, i_h, i_w, 1, 1]), [batch*i_c, 1, 1, f_h, f_w]) # coords_h [batch*i_c, i_h, i_w, f_h, f_w) 
    coord_w = tf.tile(tf.reshape(coord_w, [1, i_h, i_w, 1, 1]), [batch*i_c, 1, 1, f_h, f_w]) # coords_w [batch*i_c, i_h, i_w, f_h, f_w) 

    coord_fh = tf.tile(tf.reshape(coord_fh, [1, 1, 1, f_h, f_w]), [batch*i_c, i_h, i_w, 1, 1]) # coords_fh [batch*i_c, i_h, i_w, f_h, f_w) 
    coord_fw = tf.tile(tf.reshape(coord_fw, [1, 1, 1, f_h, f_w]), [batch*i_c, i_h, i_w, 1, 1]) # coords_fw [batch*i_c, i_h, i_w, f_h, f_w) 

    coord_h = coord_h + coord_fh + offset_map_h
    coord_w = coord_w + coord_fw + offset_map_w
    coord_h = tf.clip_by_value(coord_h, clip_value_min = 0, clip_value_max = i_h-1) # [batch*i_c, i_h, i_w, f_h, f_w]
    coord_w = tf.clip_by_value(coord_w, clip_value_min = 0, clip_value_max = i_w-1) # [batch*i_c, i_h, i_w, f_h, f_w]

    coord_hm = tf.cast(tf.floor(coord_h), tf.int32) # [batch*i_c, i_h, i_w, f_h, f_w]
    coord_hM = tf.cast(tf.ceil(coord_h), tf.int32) # [batch*i_c, i_h, i_w, f_h, f_w]
    coord_wm = tf.cast(tf.floor(coord_w), tf.int32) # [batch*i_c, i_h, i_w, f_h, f_w]
    coord_wM = tf.cast(tf.ceil(coord_w), tf.int32) # [batch*i_c, i_h, i_w, f_h, f_w]

    x_r = tf.reshape(tf.transpose(input2, [3, 0, 1, 2]), [-1, i_h, i_w]) # [i_c*batch, i_h, i_w]

    bc_index= tf.tile(tf.reshape(tf.range(batch*i_c), [-1,1,1,1,1]), [1, i_h, i_w, f_h, f_w])

    coord_hmwm = tf.concat(values=[tf.expand_dims(bc_index,-1), tf.expand_dims(coord_hm,-1), tf.expand_dims(coord_wm,-1)] , axis=-1) # [batch*i_c, i_h, i_w, f_h, f_w, 3] (batch*i_c, coord_hm, coord_wm)
    coord_hmwM = tf.concat(values=[tf.expand_dims(bc_index,-1), tf.expand_dims(coord_hm,-1), tf.expand_dims(coord_wM,-1)] , axis=-1) # [batch*i_c, i_h, i_w, f_h, f_w, 3] (batch*i_c, coord_hm, coord_wM)
    coord_hMwm = tf.concat(values=[tf.expand_dims(bc_index,-1), tf.expand_dims(coord_hM,-1), tf.expand_dims(coord_wm,-1)] , axis=-1) # [batch*i_c, i_h, i_w, f_h, f_w, 3] (batch*i_c, coord_hM, coord_wm)
    coord_hMwM = tf.concat(values=[tf.expand_dims(bc_index,-1), tf.expand_dims(coord_hM,-1), tf.expand_dims(coord_wM,-1)] , axis=-1) # [batch*i_c, i_h, i_w, f_h, f_w, 3] (batch*i_c, coord_hM, coord_wM)

    var_hmwm = tf.gather_nd(x_r, coord_hmwm) # [batch*ic, i_h, i_w, f_h, f_w]
    var_hmwM = tf.gather_nd(x_r, coord_hmwM) # [batch*ic, i_h, i_w, f_h, f_w]
    var_hMwm = tf.gather_nd(x_r, coord_hMwm) # [batch*ic, i_h, i_w, f_h, f_w]
    var_hMwM = tf.gather_nd(x_r, coord_hMwM) # [batch*ic, i_h, i_w, f_h, f_w]

    coord_hm = tf.cast(coord_hm, tf.float32) 
    coord_hM = tf.cast(coord_hM, tf.float32) 
    coord_wm = tf.cast(coord_wm, tf.float32)
    coord_wM = tf.cast(coord_wM, tf.float32)

    x_ip = var_hmwm*(coord_hM-coord_h)*(coord_wM-coord_w) + \
           var_hmwM*(coord_hM-coord_h)*(1-coord_wM+coord_w) + \
           var_hMwm*(1-coord_hM+coord_h)*(coord_wM-coord_w) + \
            var_hMwM*(1-coord_hM+coord_h)*(1-coord_wM+coord_w) # [batch*ic, ih, i_w, f_h, f_w]
    x_ip = tf.transpose(tf.reshape(x_ip, [i_c, batch, i_h, i_w, f_h, f_w]), [1,2,4,3,5,0]) # [batch, i_h, f_h, i_w, f_w, i_c]
    x_ip = tf.reshape(x_ip, [batch, i_h*f_h, i_w*f_w, i_c]) # [batch, i_h*f_h, i_w*f_w, i_c]
    with tf.variable_scope(scope or "deform_conv"):
        #deform_conv = conv2d(x_ip, filter_shape, strides=[1, f_h, f_w, 1], activation=activation, scope="deform_conv")
        deform_conv = adaptive_conv(x_ip, kernel_size=f_h, stride=f_h, output_channals=num_outputs, mode="feature", reuse=reuse)
    return deform_conv, offset'''
##################################################################################
# Residual-block
##################################################################################

def resblock(x_init, channels, use_bias=True, is_training=True, sn=False, scope='resblock'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
            x = batch_norm(x, is_training)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
            x = batch_norm(x, is_training)

        return x + x_init
    

##################################################################################
# Sampling
##################################################################################

def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2])

    return gap

def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [np.int32(h * scale_factor), np.int32(w * scale_factor)]
    return tf.image.resize_nearest_neighbor(x, size=new_size)

def up_sample_bilinear(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [np.int32(h * scale_factor), np.int32(w * scale_factor)]
    return tf.image.resize_bilinear(x, size=new_size)

def up_sample_bicubic(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [np.int32(h * scale_factor), np.int32(w * scale_factor)]
    return tf.image.resize_bicubic(x, size=new_size)
##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.tanh(x)

##################################################################################
# Normalization function
##################################################################################

def batch_norm(x, is_training=True, scope='batch_norm'):
    #return tf.layers.batch_normalization(x, training=is_training)
    return tf_contrib.layers.batch_norm(x,decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=tf.GraphKeys.UPDATE_OPS,
                                        is_training=is_training, scope=scope)

def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

##################################################################################
# Loss function
##################################################################################

def class_loss(class_logits, label, num_class):
    loss = 0
    loss = tf.losses.softmax_cross_entropy(tf.one_hot(label, num_class), class_logits, weights=1.0)

    return loss

def discriminator_loss(loss_func, real, fake, realf, fakef):
    real_loss = 0
    fake_loss = 0

    if loss_func.__contains__('wgan') :
        #real_loss = -tf.reduce_mean(realf-fakef)
        #fake_loss = 0
        real_loss = -tf.reduce_mean(real)
        fake_loss = tf.reduce_mean(fake)
        
    if loss_func == 'lsgan' :
        real_loss = tf.reduce_mean(tf.squared_difference(real, 1.0))
        fake_loss = tf.reduce_mean(tf.square(fake))

    if loss_func == 'gan' or loss_func == 'dragan' :
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake))

    if loss_func == 'hinge' :
        real_loss = tf.reduce_mean(relu(1.0 - real))
        fake_loss = tf.reduce_mean(relu(1.0 + fake))

    loss = real_loss + fake_loss

    return loss

def generator_loss(loss_func, fake, fakef):
    fake_loss = 0

    if loss_func.__contains__('wgan') :
        #fake_loss = -tf.reduce_mean(fakef)
        fake_loss = -tf.reduce_mean(fake)

    if loss_func == 'lsgan' :
        fake_loss = tf.reduce_mean(tf.squared_difference(fake, 1.0))

    if loss_func == 'gan' or loss_func == 'dragan' :
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake))

    if loss_func == 'hinge' :
        fake_loss = -tf.reduce_mean(fake)

    loss = fake_loss

    return loss

def encoder_loss(loss_func, real, realf):
    real_loss = 0

    if loss_func.__contains__('wgan') :
        #real_loss = tf.reduce_mean(realf)
        real_loss = tf.reduce_mean(real)

    if loss_func == 'lsgan' :
        real_loss = tf.reduce_mean(tf.square(real))

    if loss_func == 'gan' or loss_func == 'dragan' :
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(real), logits=real))

    if loss_func == 'hinge' :
        real_loss = tf.reduce_mean(real)

    loss = real_loss

    return loss
