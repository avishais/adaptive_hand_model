""" 
Author: Avishai Sintov
"""
from __future__ import division, print_function, absolute_import

from nn_functions import * # My utility functions

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.io import loadmat
import time


class predict_nn:
    
    def __init__(self):

        model_file = "./models/cp_8.ckpt"

        # Network Parameters
        num_input = 8
        num_output = 6
        hidden_layers = [72]*2
        activation = 2

        x_mu = np.array([ 7.53983535e+01, -3.38066395e+02,  5.28607692e-01,  4.26463315e-01, 1.02009669e+02, -9.61020193e+01,  9.18136446e-03,  8.20907553e-03, -4.26041625e-03,  3.07213955e-02,  3.01654645e-05,  2.67301021e-05, 7.55110990e-03, -7.53022741e-03])
        x_sigma = np.array([1.20656722e+02, 5.48090069e+01, 6.56873644e-02, 6.43842025e-02, 9.15623420e+01, 8.60852468e+01, 5.92933601e-02, 5.94357727e-02, 3.76585272e-01, 1.58610485e-01, 3.96773340e-04, 1.80385971e-03, 5.84777356e+00, 5.63405418e+00])

        # tf Graph input 
        X = tf.placeholder("float", [None, num_input])
        Y = tf.placeholder("float", [None, num_output])

        # Store layers weight & bias
        weights, biases = wNb(num_input, hidden_layers, num_output)

        prediction = neural_net(X, weights, biases, activation)

        # cost = tf.reduce_mean(0.5*tf.pow(prediction - Y, 2))#/(2*n)

        sess = tf.Session()

        # Restore variables from disk.
        saver = tf.train.Saver()
        saver.restore(sess, model_file)

    def predict(self, sa):
        
        sa = normzG(sa, self.x_mu, self.x_sigma)

        s_next = self.sess.run(self.prediction, {X: sa})

        s_next = denormzG(s_next, self.x_mu[8:], self.x_sigma[8:])

        return s_next

        
if __name__ == "__main__":
    NN = predict_nn()

    sa = [1.16131130e+02, -4.09232344e+02,  4.59418000e-01,  3.72727000e-01, 4.40000000e+01, -4.40000000e+01,  6.00000000e-02, -6.00000000e-02]
    





