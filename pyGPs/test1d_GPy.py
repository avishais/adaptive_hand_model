import GPy

import numpy as np
from matplotlib import pyplot as plt


# GPy.plotting.change_plotting_library('plotly')

X = np.random.uniform(-3.,3.,(20,1))
Y = np.sin(X) + np.random.randn(20,1)*0.05

Xt = np.linspace(-3.,3.,100).reshape((100,1))

print(X.shape,Y.shape)

kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
m = GPy.models.GPRegression(X,Y,kernel)

m.optimize(messages=True)
m.optimize_restarts(num_restarts = 10)

Yt, sigma = m.predict(Xt)

# print(sigma)

fig = plt.figure()
plt.plot(Xt, Yt, 'bo', label=u'Prediction')
plt.plot(X, Y, 'r.', markersize=10, label=u'Observations')
plt.fill(np.concatenate([Xt, Xt[::-1]]),
         np.concatenate([Yt - 10.9600 * sigma,
                        (Yt + 10.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
# plt.ylim(-10, 20)
plt.legend(loc='upper left')

plt.show()

