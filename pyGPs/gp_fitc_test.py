from __future__ import division, print_function, absolute_import
from past.utils import old_div

import numpy as np
import pyGPs
from sklearn.neighbors import KDTree #pip install -U scikit-learn
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pickle

import logging
logging.basicConfig()

K = 100 # Number of NN

saved = False

mode = 8
Q = loadmat('../data/data_25_' + str(mode) + '.mat')
Qtrain = Q['Xtraining']
Qtest = Q['Xtest']
Qtest = Qtest[:180,:]

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

Xtrain = Qtrain[:,0:state_action_dim]
Ytrain = Qtrain[:,state_action_dim:]
Xtest = Qtest[:,0:state_action_dim]
Ytest = Qtest[:,state_action_dim:]

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
    Xtest[:,i] = (Xtest[:,i]-x_min_X[i])/(x_max_X[i]-x_min_X[i])
for i in range(Ytrain.shape[1]):
    Ytrain[:,i] = (Ytrain[:,i]-x_min_Y[i])/(x_max_Y[i]-x_min_Y[i])
    Ytest[:,i] = (Ytest[:,i]-x_min_Y[i])/(x_max_Y[i]-x_min_Y[i])


W = np.concatenate( ( np.array([np.sqrt(3.), np.sqrt(3.)]).reshape(1,2), np.ones((1,state_dim)) ), axis=1 ).T
W = W.reshape((W.shape[0],))

if not saved:
    print("Loading data to kd-tree...")
    Xtrain_nn = Xtrain * W
    kdt = KDTree(Xtrain_nn, leaf_size=10, metric='euclidean')

#######

def predict(query):
    # idx = kdt.query(sa.T * W, k=K, return_distance=False)

    idx = kdt.query_radius(sa.T * W, r=0.15)
    K = len(idx[0])
    idx = idx[0].reshape((1,K))
 
    X_nn = Xtrain[idx,:].reshape(K, state_action_dim)
    Y_nn = Ytrain[idx,:].reshape(K, state_dim)

    y_pred = np.zeros(state_dim)
    for dim in range(state_dim):
        model = pyGPs.GPR_FITC()      # specify model (GP regression)
        
        num_u = np.fix(10)
        u = np.tile(np.linspace(0,1,num_u).T, (1, state_action_dim))
        u = np.reshape(u,(int(num_u),state_action_dim))

        # Nu = 10
        # u = np.random.rand(Nu, state_action_dim)

        m = pyGPs.mean.Linear( D=X_nn.shape[1] )
        k = pyGPs.cov.RBF(log_ell=5., log_sigma=-5)
        # k1 = pyGPs.cov.RBF(np.log(67.), np.log(66.))
        # k2 = pyGPs.cov.Periodic(np.log(1.3), np.log(1.0), np.log(2.4)) * pyGPs.cov.RBF(np.log(90.), np.log(2.4))
        # k3 = pyGPs.cov.RQ(np.log(1.2), np.log(0.66), np.log(0.78))
        # k4 = pyGPs.cov.RBF(np.log(old_div(1.6,12.)), np.log(0.18)) + pyGPs.cov.Noise(np.log(0.19))
        # k  = k1 + k2 + k3 + k4
        model.setPrior(mean=m, kernel=k, inducing_points=u) 

        model.setData(X_nn, Y_nn[:,dim]) # fit default model (mean zero & rbf kernel) with data
        model.getPosterior()
        model.optimize(X_nn, Y_nn[:,dim])     # optimize hyperparamters (default optimizer: single run minimize)

        model.predict(sa.reshape(1,state_action_dim))         # predict test cases
        y_pred[dim] = model.ym

    return y_pred, X_nn, Y_nn


# print(Xtest[170:180,state_dim:state_action_dim])
# s = np.array([0.48232082, 0.2534946 , 0.74887955, 0.70893628, 0.64893844, 0.86339834]) # 173
s = np.array([0.4816169,  0.25428246, 0.74912899, 0.7094346,  0.64462934, 0.86463593]) #174
a = Xtest[174,state_dim:state_action_dim]
sa = np.concatenate((s,a)).reshape(-1,1)
s_next, X_nn, Y_nn = predict(sa)

def Action(a):
    b = 100
    if a[0]==0 and a[1]==0:
        return np.array([0, -1])/b
    if a[0]==1 and a[1]==1:
        return np.array([0, 1])/b
    if a[0]==1 and a[1]==0:
        return np.array([-1, 0])/b
    if a[0]==0 and a[1]==1:
        return np.array([1, 0])/b


plt.figure(1)
# plt.plot(Xtest[170:180,0], Xtest[170:180,1], 'k.-')
plt.plot(np.array([s[0], s_next[0]]), np.array([s[1], s_next[1]]), 'r.-')
plt.plot(s[0], s[1], 'ko-')
plt.plot(X_nn[:,0],X_nn[:,1], '.b')
for i in range(X_nn.shape[0]-1):
    a = X_nn[i,state_dim:state_action_dim]
    a = Action(a)
    plt.arrow(X_nn[i,0],X_nn[i,1],a[0], a[1])

print(s)
print(s_next)

plt.show()



# Ypred
# [[0.48364569 0.25166836 0.74890212 0.70639912 0.65672628 0.86405539] #170
#  [0.4831057  0.25198454 0.74974338 0.70599826 0.65333641 0.86447469]
#  [0.48232372 0.25230764 0.75012168 0.71189113 0.65182271 0.8652735 ]
#  [0.48232082 0.2534946  0.74887955 0.70893628 0.64893844 0.86339834]
#  [0.4816169  0.25428246 0.74912899 0.7094346  0.64462934 0.86463593] #174
#  [0.48982902 0.27224401 0.72785797 0.69430163 0.59900633 0.86527903]
#  [0.48696039 0.27569339 0.72974499 0.69506306 0.57403011 0.86370339]
#  [0.48525589 0.28231259 0.73030183 0.69838401 0.56255913 0.86549884]
#  [0.48287428 0.28265431 0.72912038 0.70181762 0.49613564 0.87309222]
#  [0.48278836 0.28379972 0.73076994 0.70045626 0.50249483 0.88785923]]

# Xtest
# [0.4873894  0.27369838 0.69141861 0.69586023 0.61143984 0.91044776 1.         0.        ]