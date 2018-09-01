from __future__ import division, print_function, absolute_import

from nn_functions import * # My utility functions

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time
import random
from scipy.io import loadmat

mode = 8

def run_net(Xt, Xtest, learning_rate, num_hidden_layer, k, activation):

    if mode==1:
        num_input = 4 
        num_output = 2
    if mode==2:
        num_input = 8 
        num_output = 6
    if mode==3:
        num_input = 12 
        num_output = 10
    if mode==4:
        num_input = 6 
        num_output = 4
    if mode==5:
        num_input = 6 
        num_output = 4
    if mode==6:
        num_input = 14 
        num_output = 12
    if mode==7:
        num_input = 16
        num_output = 14
    if mode==8:
        num_input = 8
        num_output = 6

    i_test_start = Xt.shape[0]
    Xt = np.concatenate((Xt, Xtest), axis=0)
    i_test_end = Xt.shape[0]
    n_test = i_test_end-i_test_start+1
    n = Xt.shape[0]-n_test

    prev_states = Xt[:,0:num_input-2]
    next_states = Xt[:,num_input:]
    actions = Xt[:, num_input-2:num_input]

    X = np.concatenate((prev_states, actions, next_states-prev_states), axis=1)

    x_max = np.max(X, axis=0)
    x_min = np.min(X, axis=0)
    x_mu = np.mean(X, axis=0)
    x_sigma = np.std(X, axis=0)
    # X = normz(X, x_max, x_min)
    X = normzG(X, x_mu, x_sigma)

    x_train = X[:,0:num_input]
    y_train = X[:,num_input:]
    x_train = np.delete(x_train, range(i_test_start, i_test_end+1), 0)
    y_train = np.delete(y_train, range(i_test_start, i_test_end+1), 0)
    x_test = X[i_test_start:i_test_end+1,0:num_input]
    y_test = X[i_test_start:i_test_end+1,num_input:]

    # Training Parameters
    num_steps = 50000
    batch_size = 150
    display_step = 100

    # Network Parameters
    hidden_layers = [k]*num_hidden_layer

    # tf Graph input 
    X = tf.placeholder("float", [None, num_input])
    Y = tf.placeholder("float", [None, num_output])

    # Store layers weight & bias
    weights, biases = wNb(num_input, hidden_layers, num_output)

    # Construct model
    prediction = neural_net(X, weights, biases, activation)

    # Define loss and optimizer, minimize the squared error
    cost = tf.reduce_mean(0.5*tf.pow(prediction - Y, 2))#/(2*n)
    # cost = tf.reduce_mean(np.absolute(y_true - y_pred))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(cost)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Start Training
    # Start a new TF session
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        # Training
        for i in range(1, num_steps+1):
            # Get the next batch 
            batch_x, batch_y = next_batch(batch_size, x_train, y_train)

            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, cost], feed_dict={X: batch_x, Y: batch_y})
            # Display logs per step
            if i % display_step == 0 or i == 1:
                print('Step %i: Minibatch Loss: %f' % (i, c))

        # Save the variables to disk.
        saver.save(sess, "./models/hcp.ckpt")

        print("Optimization Finished!")

        # Testing
        # Calculate cost for training data
        y_train_pred = sess.run(prediction, {X: x_train})
        training_cost = sess.run(cost, feed_dict={X: x_train, Y: y_train})
        print("Training cost:", training_cost)

        y_test_pred = sess.run(prediction, {X: x_test})
        testing_cost = sess.run(cost, feed_dict={X: x_test, Y: y_test})
        print("Testing cost=", testing_cost)
        
        return training_cost, testing_cost

def log2file(i, learning_rate, num_hidden_layer, k, activation, training_cost, testing_cost):
    f = open('./models/hparam.txt','a+')

    # f.write("-----------------------------\n")
    # f.write("Trial %d\n" % i)
    # f.write("Learning rate: %f\n" % learning_rate)
    # f.write("Number of hidden layers: %d\n" % num_hidden_layer)
    # f.write("Number of neurons in each layer: %d\n" % k)
    # f.write("Activation: %d" % activation)
    # f.write("Training cost: %f\n" % training_cost)
    # f.write("Testing cost: %f\n" % testing_cost)

    f.write("%d %.5f %d %d %d, %f %f\n" % (i, learning_rate, num_hidden_layer, k, activation, training_cost, testing_cost))

    f.close()


def main():

    Q = loadmat('./data/Ca_20_' + str(mode) + '.mat')
    X = Q['Xtraining']
    Xtest = Q['Xtest1']['data'][0][0]

    lr = [0.0001, 0.0005,	0.001,	0.002,	0.003,	0.004,	0.005,	0.006,	0.007,	0.008,	0.009,	0.01,	0.02,	0.0288888888888889,	0.0377777777777778,	0.0466666666666667,	0.0555555555555556,	0.0644444444444444,	0.0733333333333333,	0.0822222222222222,	0.0911111111111111,	0.1,	0.110000000000000,	0.120000000000000]

    for i in range(200):
        learning_rate = lr[random.randint(0,len(lr)-1)]
        num_hidden_layer = random.randint(1,7)
        size_layers = random.randint(6,200)
        activation = 2#random.randint(1,3)

        print("-----------------------------")
        print("Trial ", i)
        print("Learning rate: ", learning_rate)
        print("Number of hidden layers: ", num_hidden_layer)
        print("Number of neurons in each layer: ", size_layers)
        if activation==1 : 
            Astr = "sigmoid" 
        elif activation==2 : 
            Astr = "relu" 
        else: Astr = "tanh"
        print("Activation function: ", Astr)

        training_cost, testing_cost = run_net(X, Xtest, learning_rate, num_hidden_layer, size_layers, activation)
        print("Training cost: ", training_cost)
        print("Testing cost: ", testing_cost)

        log2file(i, learning_rate, num_hidden_layer, size_layers, activation, training_cost, testing_cost)

if __name__ == '__main__':
  main()