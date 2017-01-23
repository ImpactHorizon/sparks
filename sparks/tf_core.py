import gzip
import os
import re
import sys
import tensorflow as tf
from sparks import tf_input

IMAGE_SIZE = tf_input.IMAGE_SIZE
NUM_CLASSES = tf_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = tf_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = tf_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 500.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

TOWER_NAME = 'tower'

def _activation_summary(x):
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _variable_on_cpu(name, shape, initializer, use_fp16=False):
    with tf.device('/cpu:0'):
        dtype = tf.float16 if use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var

def _variable_with_weight_decay(name, shape, stddev, wd, use_fp16=False):
    dtype = tf.float16 if use_fp16 else tf.float32
    var = _variable_on_cpu(
                    name,
                    shape,
                    tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def distorted_inputs(data_dir, leave_idx, batch_size=256, use_fp16=False):
    data_dir = os.path.join(data_dir, '')
    images, labels = tf_input.distorted_inputs(data_dir=data_dir,
                                                  batch_size=batch_size,
                                                  leave_idx=leave_idx)
    if use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels

def inputs(data_dir, use_fold, batch_size=256, use_fp16=False):
    data_dir = os.path.join(data_dir, '')
    images, labels = tf_input.inputs(data_dir=data_dir,
                                        use_fold=use_fold,
                                        batch_size=batch_size)
    if use_fp16:
        images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.int32)
    return images, labels

def inputs_from_slide(filename, batch_size):
    images = tf_input.inputs_from_slide(filename, batch_size)
    return images

def inference(images, batch_size=256):
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                                shape=[5, 5, 3, 32],
                                                stddev=5e-2,
                                                wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv1)

    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                            padding='SAME', name='pool1')
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                        name='norm1')

    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                                shape=[5, 5, 32, 32],
                                                stddev=5e-2,
                                                wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv2)

    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                        name='norm2')
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                            strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, [batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, 192],
                                                stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, 
                            name=scope.name)
        _activation_summary(local3)

    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[192, 96],
                                                stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [96], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, 
                            name=scope.name)
        _activation_summary(local4)

    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [96, NUM_CLASSES],
                                          stddev=1/96.0, wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, 
                                name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear


def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)  
    batch_size = logits.get_shape()[0].value  
    weights = tf.constant(batch_size*[1, 20], tf.float32, 
                            shape=logits.get_shape())
    softmax = tf.nn.softmax(logits)
    softmax = tf.clip_by_value(softmax, 1e-10, 0.999999)
    
    with tf.device('/cpu:0'):
        targets = tf.one_hot(labels, depth=2)
    
    cross_entropy = -tf.reduce_mean(weights*targets*tf.log(softmax) + 
                                        weights*(1-targets)*tf.log(1-softmax), 
                                        reduction_indices=[1])    
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name +' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op

def train(total_loss, global_step, batch_size=256):
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    loss_averages_op = _add_loss_summaries(total_loss)

    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op
