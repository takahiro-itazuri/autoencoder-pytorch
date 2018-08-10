import os
import numpy as np
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from model import Autoencoder

def to_img(x):
  x = 0.5 * (x + 1) # [-1, 1] -> [0, 1]
  x = x.clamp(0, 1)
  x = x.view(x.size(0), 1, 28, 28)
  return x

use_gpu = torch.cuda.is_available()
num_epochs = 100
batch_size = 128
learning_rate = 1e-3
log_dir = 'logs'
data_dir = 'data'

if not os.path.exists(log_dir):
  os.makedirs(log_dir)

if not os.path.exists(data_dir):
  os.makedirs(data_dir)

img_transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset = MNIST(data_dir, train=True, download=True, transform=img_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = MNIST(data_dir, train=False, download=True, transform=img_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model = Autoencoder()
if use_gpu:
  model.cuda()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

loss_list = []
test_loss_list = []

for epoch in range(num_epochs+1):
  # train
  train_loss = 0
  num_iters = 0
  for img, _ in train_loader:
    # batch_size x 784
    x = img.view(img.size(0), -1)

    if use_gpu:
      x = Variable(x).cuda()
    else:
      x = Variable(x)
    
    xhat = model(x)

    loss = criterion(xhat, x)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_loss += loss.item()
    num_iters += 1

  train_loss /= num_iters
  loss_list.append(train_loss)

  # test
  test_loss = 0
  num_iters = 0
  for img, _ in test_loader:
    x = img.view(img.size(0), -1)

    if use_gpu:
      x = Variable(x).cuda()
    else:
      x = Variable(x)
    
    xhat = model(x)

    loss = criterion(xhat, x)
    test_loss += loss.item()
    num_iters += 1
  
  test_loss /= num_iters
  test_loss_list.append(test_loss)

  print('epoch [{}/{}], loss: {:.4f}, test_loss: {:.4f}'.format(epoch, num_epochs, train_loss, test_loss))

  if epoch % 10 == 0:
    img = to_img(xhat.cpu().data)
    save_image(img, './{}/img_{}.png'.format(log_dir, epoch))

# visualize loss
plt.plot(loss_list, label='loss')
plt.plot(test_loss_list, label='test_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.savefig('./{}/loss.png'.format(log_dir))

# visualize latent space
test_dataset = MNIST(data_dir, download=True, train=False, transform=img_transform)
test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=False)

x, labels = iter(test_loader).next()
x = x.view(x.size(0), -1)

if use_gpu:
  x = Variable(x).cuda()
  z = model.encoder(x).cpu().data.numpy()
else:
  x = Variable(x)
  z = model.encoder(x).data.numpy()

plt.figure(figsize=(10, 10))
plt.scatter(z[:, 0], z[:, 1], marker='.', c=labels.numpy(), cmap=plt.cm.jet)
plt.colorbar()
plt.grid()
plt.savefig('./{}/latent_space.png'.format(log_dir))

# save result
np.save('./{}/loss_list.npy'.format(log_dir), np.array(loss_list))
torch.save(model.state_dict(), './{}/model_weights.pth'.format(log_dir))