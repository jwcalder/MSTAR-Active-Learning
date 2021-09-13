#Code for training CNN and saving CNN model
#The variable train_fraction_list controls how much training data
#to use. The models are saved in the directory models/ 
#with names indicating the amount of training data used. 
#The indices of the training data are saved in models/ as well.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import StepLR
import sys
import os

#Imports from our package
import utils
import models

#Training settings
train_fraction_list = np.arange(0.05,1.05,0.05) #List of fractions of full training data to use
cuda = True   #Use GPU acceleration
batch_size = 150
learning_rate = 1    #Learning rate
gamma = 0.9     #Learning rate step
epochs = 50

def train(model, device, data_train, target_train, optimizer, epoch, batch_size):

    model.train()
    batch_idx = 0
    for idx in range(0,len(data_train),batch_size):
        data, target = data_train[idx:idx+batch_size], target_train[idx:idx+batch_size]
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        #If by chance we get a batch size of 1, batchnormalization fails, so let's just skip
        if data.shape[0] > 1:
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


#Load data and stack mag,real phase, and imaginary phase together
hdr, fields, mag, phase = utils.load_MSTAR()
data = utils.polar_transform(mag, phase)
data = torch.from_numpy(data).float()

#Convert target names to integer labels
labels, target_names = utils.targets_to_labels(hdr)
labels = torch.from_numpy(labels).long()

#Cuda and model to device
use_cuda = cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = models.CNN().to(device)


#Loop over different amounts of training data
for train_fraction in train_fraction_list:


    #Training and testing split
    full_train_mask, test_mask, train_idx = utils.train_test_split(hdr,train_fraction)
    data_train = data[train_idx,:,:,:]
    data_test = data[test_mask,:,:,:]
    target_train = labels[train_idx]
    target_test = labels[test_mask]

    #Randomly shuffle training data
    P = np.random.permutation(len(train_idx))
    data_train = data_train[P,:,:,:]
    target_train = target_train[P]

    #Reset parameters in the model
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

    #Reset optimizer and scheduler
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    #Open results file to write accuracy
    f = open(os.path.join('results','SAR10_CNN_%d_accuracy.csv'%len(train_idx)),"w")
    f.write('Epoch,Test Accuracy,Train Accuracy\n')

    #Main training loop
    for epoch in range(1, epochs + 1):
        train(model, device, data_train, target_train, optimizer, epoch, batch_size)
        print('\nEpoch: %d'%epoch)
        test_acc = test(model, device, data_test, target_test, 'Test ')
        train_acc = test(model, device, data_train, target_train, 'Train')
        f.write("%d,%.2f,%.2f\n"%(epoch,test_acc,train_acc))
        scheduler.step()

    
    #Save model
    torch.save(model, os.path.join('models','SAR10_CNN_%d.pt'%len(train_idx)))

    #Save specific training indices used to train the CNN
    np.save(os.path.join('models',"SAR10_CNN_%d_training_indices"%len(train_idx)), train_idx)



