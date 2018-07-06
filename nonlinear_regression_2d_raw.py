""" Neural Network.

A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with TensorFlow. 


Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time

def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    Similar to mnist.train.next_batch(num)
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def gen_data(n):
    # Create data

    x = np.random.sample((n,1))#*10 - 5
    # y = x * x + 1 * x
    # y = x**5 - 10*x**3 - 20*x**2 - 1505*x - 7412;
    # y = (x+3)*(x+1)*(x-2)
    y = 0.2+0.4*x**2+0.3*x*np.sin(15*x)+0.05*np.cos(50*x)

    x = (x - np.min(x))/(np.max(x)-np.min(x))
    y = (y - np.min(y))/(np.max(y)-np.min(y))

    s = 0.1
    # y += np.random.sample((n,1))*s - s/2 
    y += np.random.normal(0, s, (n,1)) 

    return x, y

n = 50000
x_train, y_train = gen_data(n)

plt.figure(0)
plt.plot(x_train, y_train, 'ro', label='Original class 0')

x_test, y_test = gen_data(100)
# plt.show()
# exit()

# Parameters
learning_rate = 0.01
num_steps = 1000
batch_size = 200
display_step = 100

# Network Parameters
n_hidden_1 = 20 # 1st layer number of neurons
n_hidden_2 = 25 # 2nd layer number of neurons
n_hidden_3 = 20 # 2nd layer number of neurons
num_input = 1
num_classes = 1 

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_3, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Create model
# ReLU is added for non-linearity
def neural_net(x):
    # Hidden fully connected layer 
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer 
    layer_2 = tf.add(tf.matmul(tf.nn.tanh(layer_1), weights['h2']), biases['b2'])
    # Hidden fully connected layer 
    layer_3 = tf.add(tf.matmul(tf.nn.tanh(layer_2), weights['h3']), biases['b3'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(tf.nn.tanh(layer_3), weights['out']) + biases['out']
    return out_layer

# Construct model
prediction = neural_net(X)

# Define loss and optimizer
cost = tf.reduce_mean(tf.pow(prediction-Y, 2))/(2*n)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
COSTS = []	# for plotting
STEPS = []	# for plotting
start = time.time()
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x, batch_y = next_batch(batch_size, x_train, y_train)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy -JUST FOR PRINTING
            c = sess.run(cost, feed_dict={X: batch_x, Y: batch_y})
            print("Step " + str(step) + ", Minibatch cost= " + "{:.4f}".format(c))

            COSTS.append(c)
            STEPS.append(step)

    print("Optimization Finished!")

    # Calculate cost for training data
    y_train_pred = sess.run(prediction, {X: x_train})
    print("Training cost:", sess.run(cost, feed_dict={X: x_train, Y: y_train}))

    y_pred = sess.run(prediction,{X: x_test})
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(prediction - Y, 2)) / (2 * x_test.shape[0]),
        feed_dict={X: x_test, Y: y_test})  # same function as cost above
    print("Testing cost=", testing_cost)
    
print('Training time: %.3f sec.' % (time.time()-start))

fig = plt.figure(1)
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(x_train, y_train, 'ro', label='Original')
ax1.plot(x_train, y_train_pred, 'go', label='Trained prediction')
ax1.plot(x_test, y_pred, 'bo', label='Tested prediction')

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(STEPS, COSTS, 'k.')
ax2.set_xlabel('Step')
ax2.set_ylabel('Cost')
ax2.set_ylim([0, np.max(COSTS)])

plt.show()
