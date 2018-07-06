""" 
Author: Avishai Sintov
"""
from __future__ import division, print_function, absolute_import

from nn_functions import * # My utility functions

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
# import scipy.io as sio
import time

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-r", help="Retrain existing model", action="store_true")
parser.add_argument("-p", help="Plot trained models", action="store_true")
args = parser.parse_args()
if args.r and args.p:
    training = True
    retrain = True
if args.r:
    training = True
    retrain = True
elif args.p:
    training = False
    retrain = False
else:
    training = True
    retrain = False

print('Loading training data...')
X = np.loadtxt('./data/data1.db')

n_test = 5000
n = X.shape[0]-n_test

x_max = np.max(X, axis=0)
x_min = np.min(X, axis=0)

X = normz(X, x_max, x_min)
x_train = X[0:n,0:4]
y_train = X[0:n,4:]
x_test = X[n:,0:4]
y_test = X[n:,4:]

# plt.figure(0)
# ax = plt.axes(projection='3d')
# ir = [0,1,2]
# ax.plot3D(x_train[:,ir[0]], x_train[:,ir[1]], x_train[:,ir[2]], 'bo')
# fig = plt.figure(2)
# plt.plot(y_train[:,0], y_train[:,1], 'ro', label='Original')
# plt.show()

# Training Parameters
learning_rate = 0.001
num_steps = int(0.5e5)
batch_size = 200
display_step = 100

# Network Parameters
hidden_layers = [10]*2
num_input = 4 
num_output = 2
activation = 1

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_output])

# Store layers weight & bias
weights, biases = wNb(num_input, hidden_layers, num_output)

# Construct model
prediction = neural_net(X, weights, biases, activation)

# Define loss 
cost = tf.reduce_mean(tf.pow(prediction - Y, 2))#/(2*n)
# cost = tf.reduce_mean(np.absolute(y_true - y_pred))
# cost = tf.reduce_sum(tf.square(y_true - y_pred))

# Define optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# optimizer = tf.train.AdagradOptimizer(learning_rate)
train_op = optimizer.minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

load_from = './models/cp.ckpt'
save_to = './models/cp.ckpt'

# Start Training
# Start a new TF session
COSTS = []	# for plotting
STEPS = []	# for plotting
start = time.time()
with tf.Session() as sess:

    if training:
    
        if  not retrain:
            # Run the initializer
            sess.run(init)
        else:
            # Restore variables from disk.
            saver.restore(sess, "./models/" + load_from)                
            # print("Loaded saved model: %s" % "./models/" + load_from)

        # Training
        for i in range(1, num_steps+1):
            # Get the next batch 
            batch_x, batch_y = next_batch(batch_size, x_train, y_train)

            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, cost], feed_dict={X: batch_x, Y: batch_y})
            # Display logs per step
            if i % display_step == 0 or i == 1:
                print('Step %i: Minibatch Loss: %f' % (i, c))
                save_path = saver.save(sess, "./models/cp_temp.ckpt")
                COSTS.append(c)
                STEPS.append(i)

        print("Optimization Finished!")

        # Save the variables to disk.
        save_path = saver.save(sess, "./models/" + save_to)
        print("Model saved in path: %s" % save_path)

        # Plot cost convergence
        plt.figure(4)
        plt.semilogy(STEPS, COSTS, 'k-')
        plt.xlabel('Step')
        plt.ylabel('Cost')
        plt.ylim([0, np.max(COSTS)])
        plt.grid(True)
    else:
        # Restore variables from disk.
        saver.restore(sess, "./models/" + load_from)

    # Testing
    # Calculate cost for training data
    y_train_pred = sess.run(prediction, {X: x_train})
    training_cost = sess.run(cost, feed_dict={X: x_train, Y: y_train})
    print("Training cost:", training_cost)

    y_test_pred = sess.run(prediction, {X: x_test})
    testing_cost = sess.run(cost, feed_dict={X: x_test, Y: y_test})
    print("Testing cost=", testing_cost)

    j = np.random.random_integers(1, x_test.shape[0])
    f = np.array(x_test[j, :]).reshape(1,4)
    yo = sess.run(prediction, {X: f})
    print("Testing point: ", f)
    print("Point ", j, ": ", yo, y_test[j,:])

    export_net(weights, biases, x_max, x_min, activation, sess, './models/net1.netxt')

# x_train = denormz(x_train, x_max, x_min)
# x_train_pred = denormz(x_train_pred, x_max, x_min)
# x_test = denormz(x_test, x_max, x_min)
# x_test_pred = denormz(x_test_pred, x_max, x_min)
# x_path = denormz(x_path, x_max, x_min)

plt.show()


