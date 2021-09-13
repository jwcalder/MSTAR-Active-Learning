#Basic script to run Laplace learning on MSTAR data at various label rates over many trials
import graphlearning as gl
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
import CNN

hdr, fields, mag, phase = utils.load_MSTAR()

## Peform polar transormation of data from Coman, Thomas.
scale_real_phase, scale_imaginary_phase = utils.polar_transform(mag, phase)
mag_data = torch.from_numpy(np.reshape(mag,(mag.shape[0],1,mag.shape[1],mag.shape[2]))).float()

#Convert targets to numerical labels
target = hdr[:,0].tolist()
classes = set(target)
label_dict = dict(zip(classes, np.arange(len(classes))))
labels = np.array([label_dict[t] for t in target],dtype=int)

#Training and testing split (based on papers)
angle = hdr[:,6].astype(int)
train_idx = angle == 17
test_idx = angle == 15


#Train testing split and convert to torch
data = torch.cat((mag_data, scale_real_phase, scale_imaginary_phase), 1)

labels = torch.from_numpy(labels).long()

data_train = data[train_idx,:,:,:]
data_test = data[test_idx,:,:,:]
target_train = labels[train_idx]
target_test = labels[test_idx]

#Randomly shuffle training data
P = torch.randperm(np.sum(train_idx))
data_train = data_train[P,:,:,:]
target_train = target_train[P]

def do_graph_learning(train_idx, target_train, train_sizes, train_size):
  """
     performs graph learning

     Inputs:
        -train_idx - training indices
        -target_train - training Labels
        -train_sizes - array of ALL different train size (% of whole dataset) to train graph learning on
        -train_size - the current size being used for training
    Returns:
        - smaller_train_size_accuracies_poisson - accuracies from poisson learning across differnet label rates
        -smaller_train_size_accuracies_laplace - accuracies from laplace learning across differnet label rates
  """
  M = np.load('SAR10_cnn.npz')
  X = M['data']   #Encoded data
  L = M['labels'] #Labels

  W = gl.knn_weight_matrix(20,data=X)

  labels = L

  target_train = np.array(target_train)

  poisson_labels = gl.graph_ssl(W,train_idx,target_train,algorithm='poisson')

  print('Using %d labels per class'%int(len(train_idx)/10))
  print('Poisson Accuracy: %.2f'%gl.accuracy(poisson_labels,labels,len(train_idx)))
  poisson_accuracies.append(gl.accuracy(poisson_labels,labels,len(train_idx)))

  laplace_labels = gl.graph_ssl(W,train_idx,target_train,algorithm='laplace')

  print('Using %d labels per class'%int(len(train_idx)/10))
  print('Laplace Accuracy: %.2f'%gl.accuracy(laplace_labels,labels,len(train_idx)))
  laplace_accuracies.append(gl.accuracy(laplace_labels,labels,len(train_idx)))

  smaller_train_size_accuracies_poisson = []
  smaller_train_size_accuracies_laplace = []

  for size in train_sizes:
    if size <= train_size:
      train_index_subset_size = int((size/train_size)*train_idx.shape[0])
      random_subset_indexes = np.random.choice(train_idx, train_index_subset_size, replace=False)
      subset_labels = labels[random_subset_indexes]

      poisson_labels = gl.graph_ssl(W,random_subset_indexes,subset_labels,algorithm='poisson')

      smaller_train_size_accuracies_poisson.append(gl.accuracy(poisson_labels,labels,len(random_subset_indexes)))

      laplace_labels = gl.graph_ssl(W,random_subset_indexes,subset_labels,algorithm='laplace')

      smaller_train_size_accuracies_laplace.append(gl.accuracy(laplace_labels,labels,len(random_subset_indexes)))

  return smaller_train_size_accuracies_poisson, smaller_train_size_accuracies_laplace

CNN_accuracies = []
laplace_accuracies = []
poisson_accuracies = []
smaller_size_accuracies_poisson = []
smaller_size_accuracies_laplace = []

train_sizes = [.4, .35, .3, .25, .2, .15, .125, .1, .075, .05, .025]

for size in train_sizes:
  print("Training with ", str(100*size), "% of the dataset")
  train_idx1, target_train, CNN_accuracy =  CNN.train_the_net(size)
  CNN_accuracies.append(CNN_accuracy)
  smaller_poisson, smaller_laplace = do_graph_learning(train_idx1, target_train, train_sizes, size)
  smaller_size_accuracies_poisson.append(smaller_poisson)
  smaller_size_accuracies_laplace.append(smaller_laplace)
  print("")
  print("")
  print("")


scaled_train_sizes = [x * 100 for x in train_sizes]

fig, ax = plt.subplots()

ax.plot(scaled_train_sizes, CNN_accuracies, '-ok')
ax.plot(scaled_train_sizes, laplace_accuracies, '-p')
ax.plot(scaled_train_sizes, poisson_accuracies, '-s')

ax.set_title("Accuracies of Classification")
ax.invert_xaxis()
#ax2.invert_yaxis()
ax.legend(["CNN", "Laplace", "Poisson"])
plt.xlabel("Training size (% of entire dataset)")
plt.ylabel("% Test Accuracy")

fig.tight_layout()
plt.show()



for i in range(len(train_sizes)-1):


  x = train_sizes[i:]

  fig, ax = plt.subplots()

  ax.plot(x, smaller_size_accuracies_laplace[i], '-ok')
  ax.plot(x, smaller_size_accuracies_poisson[i], '-p')


  ax.set_title("Accuracies of Classification with " + str(100*train_sizes[i]) + " % CNN Training Output")


  plt.xlabel("Training size (% of entire dataset)")
  plt.ylabel("% Test Accuracy")
  plt.axhline(y=CNN_accuracies[i], color='g', linestyle='--')
  ax.legend(["Laplace", "Poisson", "CNN"])

  fig.tight_layout()
  plt.show()
