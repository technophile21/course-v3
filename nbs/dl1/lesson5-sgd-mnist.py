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

# ## MNIST SGD

# Get the 'pickled' MNIST dataset from http://deeplearning.net/data/mnist/mnist.pkl.gz. We're going to treat it as a standard flat dataset with fully connected layers, rather than using a CNN.

path = Config().data_path()/'mnist'

path.ls()

with gzip.open(path/'mnist.pkl.gz', 'rb') as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')

plt.imshow(x_train[0].reshape((28,28)), cmap="gray")
x_train.shape

x_train,y_train,x_valid,y_valid = map(torch.tensor, (x_train,y_train,x_valid,y_valid))
n,c = x_train.shape
x_train.shape, y_train.min(), y_train.max()

# In lesson2-sgd we did these things ourselves:
#
# ```python
# x = torch.ones(n,2) 
# def mse(y_hat, y): return ((y_hat-y)**2).mean()
# y_hat = x@a
# ```
#
# Now instead we'll use PyTorch's functions to do it for us, and also to handle mini-batches (which we didn't do last time, since our dataset was so small).

bs=64
train_ds = TensorDataset(x_train, y_train)
valid_ds = TensorDataset(x_valid, y_valid)
data = DataBunch.create(train_ds, valid_ds, bs=bs)

x,y = next(iter(data.train_dl))
x.shape,y.shape

class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10, bias=True)

    def forward(self, xb): return self.lin(xb)

model = Mnist_Logistic() #.cuda()

model

model.lin

model(x).shape

[p.shape for p in model.parameters()]

lr=2e-2

loss_func = nn.CrossEntropyLoss()

def update(x,y,lr):
    wd = 1e-5
    y_hat = model(x)
    # weight decay
    w2 = 0.
    for p in model.parameters(): w2 += (p**2).sum()
    # add to regular loss
    loss = loss_func(y_hat, y) + w2*wd
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            p.sub_(lr * p.grad)
            p.grad.zero_()
    return loss.item()

losses = [update(x,y,lr) for x,y in data.train_dl]

plt.plot(losses);

class Mnist_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(784, 50, bias=True)
        self.lin2 = nn.Linear(50, 10, bias=True)

    def forward(self, xb):
        x = self.lin1(xb)
        x = F.relu(x)
        return self.lin2(x)

model = Mnist_NN() #"".cuda()

losses = [update(x,y,lr) for x,y in data.train_dl]

plt.plot(losses);

model = Mnist_NN() #.cuda()

def update(x,y,lr):
    opt = optim.Adam(model.parameters(), lr)
    y_hat = model(x)
    loss = loss_func(y_hat, y)
    loss.backward()
    opt.step()
    opt.zero_grad()
    return loss.item()

losses = [update(x,y,1e-3) for x,y in data.train_dl]

plt.plot(losses);

learn = Learner(data, Mnist_NN(), loss_func=loss_func, metrics=accuracy)

learn.lr_find()
learn.recorder.plot()

learn.fit_one_cycle(1, 1e-2)

learn.recorder.plot_lr(show_moms=True)

learn.recorder.plot_losses()

# ## fin


