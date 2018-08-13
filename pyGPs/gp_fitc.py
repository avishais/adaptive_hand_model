from __future__ import division, print_function, absolute_import

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

mode = 5
Q = loadmat('../data/Ca_25_' + str(mode) + '.mat')
Qtrain = Q['Xtraining']
Qtest = Q['Xtest3']['data'][0][0]
# Qtest = Qtest[:200,:]

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

# Xtrain = np.concatenate( (Xtrain, Xtest), axis=0 )
# Ytrain = np.concatenate( (Ytrain, Ytest), axis=0 )

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


W = np.concatenate( ( np.array([np.sqrt(1.), np.sqrt(1.)]).reshape(1,2), np.ones((1,state_dim)) ), axis=1 ).T
W = W.reshape((W.shape[0],))

if not saved:
    print("Loading data to kd-tree...")
    Xtrain_nn = Xtrain# * W
    kdt = KDTree(Xtrain_nn, leaf_size=10, metric='euclidean')

#######

def predict(query):
    idx = kdt.query(sa.T, k=K, return_distance=False)
    # idx = kdt.query_radius(sa.T * W, r=0.2)
    # k = len(idx[0])
    # if k > 0:
    #     idx = idx[0].reshape((1,k))
    #     if k > K:
    #         idx = idx[0,:K]
    #         k = K
    # else:
    #     idx = kdt.query(sa.T * W, k=K, return_distance=False)   
    #     k = K
    
    X_nn = Xtrain[idx,:].reshape(K, state_action_dim)
    Y_nn = Ytrain[idx,:].reshape(K, state_dim)

    y_pred = np.zeros(state_dim)
    for dim in range(state_dim):
        model = pyGPs.GPR_FITC()      # specify model (GP regression)
        
        num_u = np.fix(15)
        u = np.tile(np.linspace(0,1,num_u).T, (1, state_action_dim))
        u = np.reshape(u,(int(num_u),state_action_dim))

        # Nu = 10
        # u = np.random.rand(Nu, state_action_dim)

        m = pyGPs.mean.Linear( D=X_nn.shape[1] )
        k = pyGPs.cov.RBF(log_ell=5., log_sigma=-5)
        model.setPrior(mean=m, kernel=k, inducing_points=u) 

        model.setData(X_nn, Y_nn[:,dim]) # fit default model (mean zero & rbf kernel) with data
        model.getPosterior()
        model.optimize(X_nn, Y_nn[:,dim])     # optimize hyperparamters (default optimizer: single run minimize)

        model.predict(sa.reshape(1,state_action_dim))         # predict test cases
        y_pred[dim] = model.ym

    return y_pred

if (saved):
    print('Loading saved path...')
    # Getting back the objects:
    with open('objs.pkl') as f:  # Python 3: open(..., 'rb')
        Xtest, Ypred = pickle.load(f)    
else:
    s = Xtest[0,:state_dim]
    Ypred = s.reshape(1,state_dim)

    print("Running path...")
    for i in range(Xtest.shape[0]):
        print("Step " + str(i))
        a = Xtest[i,state_dim:state_action_dim]
        sa = np.concatenate((s,a)).reshape(-1,1)
        s_next = predict(sa)
        print(s_next)
        s = s_next
        Ypred = np.append(Ypred, s.reshape(1,state_dim), axis=0)
    
    with open('objs.pkl', 'w') as f:  # Python 3: open(..., 'wb')
        pickle.dump([Xtest, Ypred], f)


plt.figure(0)
plt.plot(Xtest[:,0], Xtest[:,1], 'k.-')
plt.plot(Ypred[:,0], Ypred[:,1], 'r.-')
# plt.ylim([0, np.max(COSTS)])
plt.axis('equal')
plt.title('pyGPs (gp_fitc.py) - ' + str(mode))
plt.grid(True)
plt.show()
