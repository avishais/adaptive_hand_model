from __future__ import division, print_function, absolute_import

import pyGPs
import numpy as np
import random

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

N = 100
x = np.random.rand(N, 1)*2 - 1
y = np.random.rand(N, 1)*2 - 1
z = 0.5*x**2 + 0.25*y**2

Nt = 200
xt = np.random.rand(Nt, 1)*4 - 2
yt = np.random.rand(Nt, 1)*4 - 2


X = np.concatenate((x,y), axis=1)
Xt = np.concatenate((xt,yt), axis=1)
Z = np.concatenate((z,z), axis=1)

print(Z,Z.shape)

model = pyGPs.GPR()      # specify model (GP regression)
model.getPosterior(X, z) # fit default model (mean zero & rbf kernel) with data
model.optimize(X, z)     # optimize hyperparamters (default optimizer: single run minimize)
model.predict(Xt)         # predict test cases
zt = model.ym

# print(zt.shape)

x  = x.reshape((N,))
y  = y.reshape((N,))
z  = z.reshape((N,))
xt  = xt.reshape((Nt,))
yt  = yt.reshape((Nt,))
zt  = zt.reshape((Nt,))


plt.figure(0)
ax = plt.axes(projection='3d')
ax.plot3D(x, y, z, 'ro',markersize=3)
ax.plot3D(xt, yt, zt, 'ko',markersize=3)

plt.show()


