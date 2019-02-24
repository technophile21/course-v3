# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %matplotlib inline
from fastai.basics import *

n = 100

data = torch.ones(n, 2)
data[:, 0].uniform_(-1., 1)
data[:5, :]

a = 12
b = 5
data[:, 1] = a*data[:, 0] + b
data[:5, :]

epochs = 2
lr = 0.1

param = torch.rand((2, 1)) # np.random.rand(2,1)
for it in range(1, epochs + 1):
    print("Epoch: ", it)
    for x in range(0, n):
        y_pred = param[0] * data[x,0] + param[1]
        err = data[x,1] - y_pred
        err2 = err*err
        print("Loss: ", err2)
        del_a = -2 * (err) * data[x, 0]
        del_b = -2 * (err)
        param[0] = param[0] - (lr * del_a)
        param[1] = param[1] - (lr * del_b)
    print("a: ", param[0], " b: ", param[1])


