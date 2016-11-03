""" MULTI ORC MODEL """
#import abc
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from mnist import read_data_sets
import tensorflow.models.image.mnist.convolutional as conv
import tensorflow as tf


#PlaceHolders

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)

def weight_variable(shape):
    """ Weight Initialization """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """ Bias Initialization """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    """ Convolution Initialization """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    """ Pooling Initialization """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')

def convolutional_layers():
    """ """
    # First Layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])


    x_image = tf.reshape(x, [-1,28,28,1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    #Second Layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    return x, h_pool2

def train_model():
    """
    Creates:
        Densely connected Layer
        Dropout
        Readout Layer

    returns x, y_conv
    """
    x, h_pool2 = convolutional_layers()

    #Densely Connected Layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    #Dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    #Readout Layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    return x, y_conv, keep_prob

# Will need use Model after
def train():
    """
    Currently takes the training data from tensorflow which
    is stored in the folder MNIST_data

    ***********
    NEED TO CHANGE MNIST.PY FILE TO INPUT OWN TRAINING IMAGES/LABELS
    PLEASE REMOVE THIS COMMENT ONCE THIS IS TAKEN CARE OF
    ***********


    Trains based off of data
    Saves checkpoint in ./train_chpt/model.ckpt
    """
    # Variables
    x, y_conv, keep_prob = train_model()
    mnist = read_data_sets('MNIST_data', one_hot=True)

    #Training prep
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Add ops to save and restore all the Variables
    saver = tf.train.Saver()
    # Runs Session
    sess = tf.InteractiveSession()
    init = tf.initialize_all_variables()
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'
    with tf.Session(config = config) as s:
        sess.run(init)

    # Runs Training Session
    for i in range(1000):
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    # Saves the variables to disk
    save_path = saver.save(sess, "train_ckpt/model.ckpt")
    print("Model saved in file: %s" % save_path)

def run(self, tests):
    pass

def main():
    train()

if __name__ == '__main__':
    main()
