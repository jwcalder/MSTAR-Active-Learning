#Code for training CNN and saving CNN model and CNN features
#Should also save the indices (or boolean mask) for the training set used to train the CNN

import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import datasets
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from scipy.special import softmax
import sys
import os

hdr, fields, mag, phase = utils.load_MSTAR()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        w = (32,64) #Number of channels in 1st and 2nd layers
        #self.conv1 = nn.Conv2d(1, w[0], 3, 1)
        self.conv1 = nn.Conv2d(3, w[0], 3, 1)
        self.conv2 = nn.Conv2d(w[0], w[1], 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        f = 512  #Number of hidden nodes in fully connected layers
        self.fc1 = nn.Linear(w[1]*10*10, f)
        self.fc2 = nn.Linear(f, 10)
        self.bn1 = nn.BatchNorm1d(f)

    def forward(self, x): #88
        x = self.conv1(x) #86
        x = F.relu(x)
        x = F.max_pool2d(x, 2) #43
        x = self.conv2(x) #41
        x = F.relu(x)
        x = F.max_pool2d(x, 4)  #10
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn1(x)   #batch normalization
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    #This is useful for extracting features from convolutional part of NN
    def convnet(self, x):
        x = self.conv1(x) #86
        x = F.relu(x)
        x = F.max_pool2d(x, 2) #43
        x = self.conv2(x) #41
        x = F.relu(x)
        x = F.max_pool2d(x, 4)  #10
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        return x

def train(model, device, data_train, target_train, optimizer, epoch, batch_size):
    model.train()
    batch_idx = 0
    for idx in range(0,len(data_train),batch_size):
        data, target = data_train[idx:idx+batch_size], target_train[idx:idx+batch_size]
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == -1:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(target_train),
                100. * batch_idx / int(len(data_train)/batch_size), loss.item()))
        batch_idx += 1


def test(model, device, data_test, target_test,name):
    batch_size = 1000
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for idx in range(0,len(data_test),batch_size):
            data, target = data_test[idx:idx+batch_size], target_test[idx:idx+batch_size]
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(target_test)

    print(name+' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(target_test),
        100. * correct / len(target_test)))

    return 100. * correct / len(target_test)





## Peform polar transormation of data from Coman, Thomas.
scale_real_phase, scale_imaginary_phase = utils.polar_transform(mag, phase)
mag_data = torch.from_numpy(np.reshape(mag,(mag.shape[0],1,mag.shape[1],mag.shape[2]))).float()

#Convert targets to numerical labels
target = hdr[:,0].tolist()
classes = set(target)
label_dict = dict(zip(classes, np.arange(len(classes))))
labels = np.array([label_dict[t] for t in target],dtype=int)


#Train testing split and convert to torch
data = torch.cat((mag_data, scale_real_phase, scale_imaginary_phase), 1)

labels = torch.from_numpy(labels).long()


def train_the_net(train_size = 0):

  #Training and testing split (based on papers)
  angle = hdr[:,6].astype(int)
  train_idx = angle == 17
  test_idx = angle == 15

  data_train = data[train_idx,:,:,:]
  data_test = data[test_idx,:,:,:]
  target_train = labels[train_idx]
  target_test = labels[test_idx]

  P = torch.randperm(np.sum(train_idx))
  data_train = data_train[P,:,:,:]
  target_train = target_train[P]

  if train_size > 0:
      all_indexes = np.arange(data.shape[0])
      train_indexes_used = all_indexes[train_idx]
      test_data_indexes = all_indexes[test_idx]
      smaller_train_indexes = np.random.choice(train_indexes_used, int(data.shape[0]*train_size), replace=False)
      smaller_data_train = data[smaller_train_indexes, :, :, :]
      smaller_target_train = labels[smaller_train_indexes]

      data_train = smaller_data_train
      target_train = smaller_target_train

      P = torch.randperm(len(smaller_train_indexes))

      data_train = data_train[P,:,:,:]
      target_train = target_train[P]
  test_accuracies = []

  #Training settings
  cuda = True   #Use GPU acceleration (Edit->Notebook Settings and enable GPU)
  batch_size = 150
  learning_rate = 1    #Learning rate
  gamma = 0.9     #Learning rate step
  epochs = 50

  #Cuda and optimizer/scheduler
  use_cuda = cuda and torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  model = Net().to(device)
  optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
  scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

  #Main training loop
  for epoch in range(1, epochs + 1):
      train(model, device, data_train, target_train, optimizer, epoch, batch_size)
      print('\nEpoch: %d'%epoch)
      test_acc = test(model, device, data_test, target_test, 'Test ')
      train_acc = test(model, device, data_train, target_train, 'Train')
      scheduler.step()
      test_accuracies.append(test_acc)

  #Save model
  torch.save(model.state_dict(), 'SAR10_cnn.pt')

  #Save encoded data
  model.eval()
  with torch.no_grad():
      Y = model.convnet(data.to(device)).cpu().numpy()
      np.savez_compressed('SAR10_cnn.npz',data=Y,labels=labels.numpy())

  return train_idx, target_train, test_accuracies[epochs-1]

train_idx, target_train, CNN_accuracy =  train_the_net()

train_idx = np.arange(data.shape[0])[train_idx]
np.save("Training indices", train_idx)
np.save("Training labels", target_train)
