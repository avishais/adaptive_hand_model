import numpy as np
from matplotlib import pyplot as plt

from sklearn.neighbors import KDTree #pip install -U scikit-learn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, RBF, ConstantKernel as C

K = 100 # Number of NN

mode = 1
Qtrain = np.loadtxt('../data/data_25_train_' + str(mode) + '.db')
Qtest = np.loadtxt('../data/data_25_test_' + str(mode) + '.db')
Qtest = Qtest[:300,:]

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

    # print(X_nn.shape, X_nn)
    # print(Y_nn.shape, Y_nn)

    y_pred = np.zeros(state_dim)
    sigma = np.zeros(state_dim)
    for dim in range(state_dim):
        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        # kernel = C(1.0, (1e-3, 1e3)) + Matern(length_scale=2, nu=3/2)

        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

        gp.fit(X_nn, Y_nn[:,dim])

        y_pred[dim], sigma[dim] = gp.predict(sa.reshape(1,state_action_dim), return_std=True)

    return y_pred

s = Xtest[0,:state_dim]
Ypred = s.reshape(1,state_dim)

# s = np.array([0.54749736, 0.18103569])
# a = Xtest[72,state_dim:state_action_dim]
# sa = np.concatenate((s,a)).reshape(-1,1)
# s_next = predict(sa)
# print(s_next)

print("Running path...")
for i in range(Xtest.shape[0]):
    print(i)
    a = Xtest[i,state_dim:state_action_dim]
    sa = np.concatenate((s,a)).reshape(-1,1)
    s_next = predict(sa)
    print(s_next)
    s = s_next
    Ypred = np.append(Ypred, s.reshape(1,state_dim), axis=0)

plt.figure(0)
plt.plot(Xtest[:,0], Xtest[:,1], 'k.-')
plt.plot(Ypred[:,0], Ypred[:,1], 'r.-')
# plt.ylim([0, np.max(COSTS)])
plt.axis('equal')
plt.title('Scikit (gp_scikit.py)')
plt.grid(True)
plt.show()

