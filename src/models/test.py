from tensorflow.examples.tutorials.mnist import input_data
import datetime
import numpy as np
import tensorflow as tf
import cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from random import randint


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder("float", [None, 784])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)
y_ = tf.placeholder("float", [None,10])

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

#Train our model
iter = 1000
for i in range(iter):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#Evaluationg our model:
correct_prediction=tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))
print "Accuracy: ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})

#1: Using our model to classify a random MNIST image from the original test set:
num = randint(0, mnist.test.images.shape[0])
img = mnist.test.images[num]

classification = sess.run(tf.argmax(y, 1), feed_dict={x: [img]})
'''
#Uncomment this part if you want to plot the classified image.
plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
plt.show()
'''
print 'Neural Network predicted', classification[0]
print 'Real label is:', np.argmax(mnist.test.labels[num])


#2: Using our model to classify MNIST digit from a custom image:

# create an an array where we can store 1 picture
images = np.zeros((1,784))
# and the correct values
correct_vals = np.zeros((1,10))

# read the image
gray = cv2.imread("my_digit.png", 0 ) #0=cv2.CV_LOAD_IMAGE_GRAYSCALE #must be .png!

# rescale it
gray = cv2.resize(255-gray, (28, 28))

# save the processed images
cv2.imwrite("my_grayscale_digit.png", gray)
"""
all images in the training set have an range from 0-1
and not from 0-255 so we divide our flatten images
(a one dimensional vector with our 784 pixels)
to use the same 0-1 based range
"""
flatten = gray.flatten() / 255.0
"""
we need to store the flatten image and generate
the correct_vals array
correct_val for a digit (9) would be
[0,0,0,0,0,0,0,0,0,1]
"""
images[0] = flatten


my_classification = sess.run(tf.argmax(y, 1), feed_dict={x: [images[0]]})

"""
we want to run the prediction and the accuracy function
using our generated arrays (images and correct_vals)
"""
print 'Neural Network predicted', my_classification[0], "for your digit"
