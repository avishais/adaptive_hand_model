import numpy as np
from matplotlib import pyplot as plt

from sklearn.neighbors import KDTree #pip install -U scikit-learn
import GPy
import time
from scipy.io import loadmat
import pickle
import math

K = 100 # Number of NN

saved = False

mode = 5
Q = loadmat('../data/Cb_20_' + str(mode) + '.mat')
Qtrain = Q['Xtraining']
Qtest = []#Q['Xtest1']['data'][0][0]
# Qtest = Qtest[1038:1038+300,:]

# Qtrain = np.loadtxt('../data/toyData.db')
# Qtest = np.loadtxt('../data/toyDataPath.db')

if mode==1:
    state_action_dim = 4 
    state_dim = 2
if mode==2:
    state_action_dim = 8 
    state_dim = 6
if mode==3:
    state_action_dim = 12 
    state_dim = 10
if mode==4:
    state_action_dim = 6 
    state_dim = 4
if mode==5:
    state_action_dim = 6 
    state_dim = 4
if mode==6:
    state_action_dim = 14 
    state_dim = 12
if mode==7:
    state_action_dim = 16
    state_dim = 14
if mode==8:
    state_action_dim = 8
    state_dim = 6
if mode==11:
    state_action_dim = 7
    state_dim = 5

Xtrain = Qtrain[:,0:state_action_dim]
Ytrain = Qtrain[:,state_action_dim:]
# Xtest = Qtest[:,0:state_action_dim]
# Ytest = Qtest[:,state_action_dim:]

# Normalize
x_max_X = np.max(Xtrain, axis=0)
x_min_X = np.min(Xtrain, axis=0)
x_max_Y = np.max(Ytrain, axis=0)
x_min_Y = np.min(Ytrain, axis=0)

for i in range(state_dim):
    tmp = np.max([x_max_X[i], x_max_Y[i]])
    x_max_X[i] = tmp
    x_max_Y[i] = tmp
    tmp = np.min([x_min_X[i], x_min_Y[i]])
    x_min_X[i] = tmp
    x_min_Y[i] = tmp

for i in range(Xtrain.shape[1]):
    Xtrain[:,i] = (Xtrain[:,i]-x_min_X[i])/(x_max_X[i]-x_min_X[i])
    # Xtest[:,i] = (Xtest[:,i]-x_min_X[i])/(x_max_X[i]-x_min_X[i])
for i in range(Ytrain.shape[1]):
    Ytrain[:,i] = (Ytrain[:,i]-x_min_Y[i])/(x_max_Y[i]-x_min_Y[i])
    # Ytest[:,i] = (Ytest[:,i]-x_min_Y[i])/(x_max_Y[i]-x_min_Y[i])


W = np.concatenate( ( np.array([np.sqrt(500.), np.sqrt(500.)]).reshape(1,2), np.ones((1,state_dim)) ), axis=1 ).T
W = W.reshape((W.shape[0],))

print("Loading data to kd-tree...")
Xtrain_nn = Xtrain * W
kdt = KDTree(Xtrain_nn, leaf_size=10, metric='euclidean')

###

def predict(sa):
    idx = kdt.query(sa.T*W, k=K, return_distance=False)
    X_nn = Xtrain[idx,:].reshape(K, state_action_dim)
    Y_nn = Ytrain[idx,:].reshape(K, state_dim)

    fig = plt.figure(1)
    plt.plot(X_nn[:,0],X_nn[:,1],'.y')
    plt.plot(sa[0],sa[1],'ob')

    print(X_nn[1:4,0:2], Y_nn[1:4,0:2])


    mu = np.zeros(state_dim)
    sigma = np.zeros(state_dim)
    for dim in range(state_dim):
        kernel = GPy.kern.RBF(input_dim=state_action_dim, variance=1., lengthscale=1.)
        m = GPy.models.GPRegression(X_nn,Y_nn[:,dim].reshape(-1,1),kernel)

        m.optimize(messages=False)
        # m.optimize_restarts(num_restarts = 10)

        mu[dim], sigma[dim] = m.predict(sa.reshape(1,state_action_dim))

    return mu, sigma

def propagate(sa):
    mu, sigma = predict(sa)
    # s_next = np.random.normal(mu, sigma, state_dim)

    return mu#s_next

# ('Current position: ', array([505., 100.]), ', waypoint: ', (483.0, 99.0), '--:')
# ('Current load: ', (14.0, -56.0))
# Current gripper pos: (0.4558190405368805, 0.37144142389297485)
# base[397 509] 0.0
# Action [0.06 0.06] predicts next state [505.34953673 100.04206298] with distance 22.37381700595351
# Action [-0.06  0.06] predicts next state [505.28970464 100.03178325] with distance 22.313572322014554
# Action [ 0.06 -0.06] predicts next state [504.4469229  100.00015542] with distance 21.470230851446452
# *** predicted next state: [504.4469229  100.00015542] with action: [ 0.06 -0.06](100) planned to have distance 21.470230851446452



obj_pos = np.array([505., 100.])
base_pos = np.array([397, 509])
base_theta = 0.
load = np.array([14.0, -56.0]).reshape((1,2))
ang = np.array([0.4558190405368805, 0.37144142389297485]).reshape((1,2))
waypoint = np.array([483.0, 99.0])

A = np.array([[0.06,0.06],[-0.06, 0.06], [0.06, -0.06],[-0.06, -0.06]])

obj_pos = obj_pos - base_pos
R = np.array([[math.cos(base_theta), -math.sin(base_theta)], [math.sin(base_theta), math.cos(base_theta)]])
obj_pos = np.matmul(R, obj_pos.T).reshape((1,2))


# for i in range(1):
a = A[1,:].reshape((1,2))
sa = np.concatenate((obj_pos, load, a), axis=1)
sa = (sa-x_min_X)/(x_max_X-x_min_X)
sa = sa.reshape(-1,1)

s_next = propagate(sa)

plt.plot(s_next[0], s_next[1],'sr')

s_next = s_next*(x_max_Y-x_min_Y) + x_min_Y
s_next = s_next[:2]

s_next = np.matmul(R.T, s_next.T)
s_next += base_pos

print(s_next.reshape((1,2)), np.linalg.norm(waypoint-s_next))

# fig = plt.figure(2)
# plt.plot(obj_pos[0], obj_pos[1],'sr')
# plt.plot(waypoint[0], waypoint[1],'sb')
# plt.axis("equal")
# # plt.axis([200, 750, 90, 350])
# plt.plot(s_next[0], s_next[1],'xg')


plt.show()