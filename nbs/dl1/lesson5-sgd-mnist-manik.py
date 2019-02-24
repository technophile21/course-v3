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

path  = Config.data_path()/'mnist'
path

path.ls()

with gzip.open(path/'mnist.pkl.gz', 'rb') as f:
    ((train_x, train_y), (val_x, val_y), _) = pickle.load(f, encoding='latin-1')
train_x.shape

x = train_x[0].reshape(28, 28)
plt.imshow(x, cmap='gray')

train_x, train_y, val_x, val_y = map(torch.tensor, (train_x, train_y, val_x, val_y))
train_x.shape

bs = 64
train_ds = TensorDataset(train_x, train_y)
val_ds = TensorDataset(val_x, val_y)
data = DataBunch.create(train_ds, val_ds, bs=bs)
data

class MNIST_Lin(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)
        
    def forward(self, x):
        return self.lin(x)

loss_func = nn.CrossEntropyLoss()

model = MNIST_Lin()

def update(x, y, lr):
    y_hat = model(x)
    wd = 1e-5  #regularization factor
    #y_hat = np.argmax(y_hat, 1)
    w = 0.
    for p in model.parameters():
        w += ((p**2).sum())
    loss = loss_func(y_hat, y) + wd * w
    
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            p.sub_(lr * p.grad)
            p.grad.zero_()
    return loss.item()

lr = 2e-2

losses = [update(x, y, lr) for x, y in data.train_dl]

plt.plot(losses)

# Using Non linear model for predicting digits

class MNIST_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(784, 50)
        self.relu1 = nn.ReLU()
        self.lin2 = nn.Linear(50, 10)
        
    def forward(self, x):
        x1 = self.lin1(x)
        x2 = self.lin2(self.relu1(x1))
        return x2

model = MNIST_NN()

losses = [update(x, y, lr) for x, y in data.train_dl]

plt.plot(losses)


