import numpy as np
from matplotlib import pyplot as plt

from sklearn.neighbors import KDTree #pip install -U scikit-learn
import GPy
import time
from scipy.io import loadmat

K = 100 # Number of NN

mode = 5
Q = loadmat('../data/data_25_' + str(mode) + '.mat')
Qtrain = Q['Xtraining']
Qtest = Q['Xtest']
# Qtest = Qtest[:500,:]

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


W = np.concatenate( ( np.array([np.sqrt(1.), np.sqrt(1.)]).reshape(1,2), np.ones((1,state_dim)) ), axis=1 ).T
W = W.reshape((W.shape[0],))

print("Loading data to kd-tree...")
Xtrain_nn = Xtrain# * W
kdt = KDTree(Xtrain_nn, leaf_size=10, metric='euclidean')

###

def predict(sa):
    idx = kdt.query(sa.T, k=K, return_distance=False)
    X_nn = Xtrain[idx,:].reshape(K, state_action_dim)
    Y_nn = Ytrain[idx,:].reshape(K, state_dim)

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
    s_next = np.random.normal(mu, sigma, state_dim)

    return s_next


start = time.time()

s = Xtest[0,:state_dim]
Ypred = s.reshape(1,state_dim)

print("Running (open loop) path...")
for i in range(Xtest.shape[0]):
    print(i)
    a = Xtest[i,state_dim:state_action_dim]
    sa = np.concatenate((s,a)).reshape(-1,1)
    # s_next = predict(sa)
    s_next = propagate(sa)
    print(s_next)
    s = s_next
    Ypred = np.append(Ypred, s.reshape(1,state_dim), axis=0)

# print("Running (closed loop) path...")
# for i in range(Xtest.shape[0]):
#     print(i)
#     s = Xtest[i,:state_dim]
#     a = Xtest[i,state_dim:state_action_dim]
#     sa = np.concatenate((s,a)).reshape(-1,1)
#     s_next = predict(sa)
#     print(s_next)
#     # s = s_next
#     Ypred = np.append(Ypred, s_next.reshape(1,state_dim), axis=0)

end = time.time()

plt.figure(0)
plt.plot(Xtest[:,0], Xtest[:,1], 'k.-')
plt.plot(Ypred[:,0], Ypred[:,1], 'r.-')
# for i in range(Ypred.shape[0]-1):
#     plt.plot(np.array([Xtest[i,0], Ypred[i,0]]), np.array([Xtest[i,1], Ypred[i,1]]), 'r.-')
# plt.ylim([0, np.max(COSTS)])
plt.axis('equal')
plt.title('GPy (gp_GPy.py) - ' + str(mode))
plt.grid(True)
plt.show()

print("Calc. time: " + str(end - start) + " sec.")