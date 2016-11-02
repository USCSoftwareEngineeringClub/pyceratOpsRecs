""" MULTI ORC MODEL """
#import abc
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.models.image.mnist.convolutional as conv
import tensorflow as tf

VALIDATION_SIZE = 5000  # Size of the validation set.
BATCH_SIZE = 64

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

    return x, y_conv

# Will need use Model after
def train(tests, ans):
    """

    Extract images/ labels form provided gz archives

    :param tests
        Name of the .gz file containing the JPEGs of the training images

    :param ans
        Name of the .gz file containing the JPEGs of the training lables

    Currently 11/1/16 Trying to figure out how to train in the session.
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/mnist/convolutional.py
    might be helpful in figuring this out

    """
    x, y_conv = train_model()
    num_epochs = 10
    train_size = train_labels.shape[0]

    # Extracts the data into numpy arrays
    train_data = extract_data(tests, tests_size)
    train_labels = extract_labels(ans, ans_size)

    # Generate a validatoin set
    validation_data = train_data[:VALIDATION_SIZE, ...]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_data = train_data[VALIDATION_SIZE:, ...]
    train_labels = train_labels[VALIDATION_SIZE:]

    #Training first steps
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Create a local session to run the Training
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'
    with tf.Session(config = config) as sess:

        # Run all the initializers to prepare the trainable parameters
        tf.initialize_all_variables().run()

        #Loop through training steps
        for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
            batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]

            feed_dict = {train_data_node: batch_data, train_labels_node: batch_labels}

            sess.run(optimizer, feed_dict=feed_dict)


def run(self, tests):
    pass

def main():
    train("../data/train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz")

if __name__ == '__main__':
    main()
