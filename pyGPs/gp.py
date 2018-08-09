from __future__ import division, print_function, absolute_import

import numpy as np
import pyGPs
from sklearn.neighbors import KDTree #pip install -U scikit-learn
import matplotlib.pyplot as plt
from scipy.io import loadmat

import logging
logging.basicConfig()

K = 100 # Number of NN

mode = 5
Q = loadmat('../data/data_25_' + str(mode) + '.mat')
Qtrain = Q['Xtraining']
Qtest = Q['Xtest']
# Qtest = Qtest[:300,:]

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


print("Loading data to kd-tree...")
kdt = KDTree(Xtrain, leaf_size=10, metric='euclidean')

#######

def predict(sa):
    idx = kdt.query(sa.T, k=K, return_distance=False)
    X_nn = Xtrain[idx,:].reshape(K, state_action_dim)
    Y_nn = Ytrain[idx,:].reshape(K, state_dim)

    y_pred = np.zeros(state_dim)
    for dim in range(state_dim):
        model = pyGPs.GPR()      # specify model (GP regression)

        m = pyGPs.mean.Linear( D=X_nn.shape[1] )# + pyGPs.mean.Const()  
        k = pyGPs.cov.RBF(log_ell=5., log_sigma=-5)
        model.setPrior(mean=m, kernel=k) 

        model.getPosterior(X_nn, Y_nn[:,dim]) # fit default model (mean zero & rbf kernel) with data
        
        rand_inx = np.random.choice(K, 10)
        model.optimize(X_nn[rand_inx,:], Y_nn[rand_inx,dim],numIterations=10)     # optimize hyperparamters (default optimizer: 40 runs minimize)
        # model.optimize(X_nn, Y_nn[:,dim])     # optimize hyperparamters (default optimizer: single run minimize)

        model.predict(sa.reshape(1,state_action_dim))         # predict test cases
        y_pred[dim] = model.ym

    return y_pred

s = Xtest[0,:state_dim]
Ypred = s.reshape(1,state_dim)

print("Running path...")
for i in range(Xtest.shape[0]):
    print(i)
    a = Xtest[i,state_dim:state_action_dim]
    sa = np.concatenate((s,a)).reshape(-1,1)
    s_next = predict(sa)
    s = s_next
    Ypred = np.append(Ypred, s.reshape(1,state_dim), axis=0)

plt.figure(0)
plt.plot(Xtest[:,0], Xtest[:,1], 'k.-')
plt.plot(Ypred[:,0], Ypred[:,1], 'r.-')
# plt.ylim([0, np.max(COSTS)])
plt.axis('equal')
plt.title('pyGPs (gp.py)')
plt.grid(True)
plt.show()
