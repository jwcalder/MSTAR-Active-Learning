#Code for running graph learning on features from
#pretrained CNN models

import numpy as np
import sys
import os
import graphlearning as gl

#Imports from our package
import utils

#Number of neighbors to use in building graph
k = 20

#Load MSTAR and CNN models
hdr, fields, mag, phase = utils.load_MSTAR()
cnn_models, cnn_train_idx, cnn_num_train =  utils.get_cnn_models()
all_train = np.load('../models/SAR10_CNN_all_train.npy')
index = np.arange(len(all_train))

#Get labels and corresponding target names
_,test_mask,_ = utils.train_test_split(hdr,1)
labels, target_names = utils.targets_to_labels(hdr)

#Open results file to write accuracy
f_cnnvae_laplace = open('../results/SAR10_CNNVAE_laplace_accuracy.csv',"w")
f_cnnvae_laplace.write('Number of Labels,Accuracy\n')
print('Number of Labels,Accuracy')

dataset = 'SAR10'
metric = 'CNNVAE'
try:
    knn_data = gl.weightmatrix.load_knn_data(dataset,metric=metric)
except:
    X = utils.encodeMSTAR('../models/SAR10_CNNVAE.pt', use_phase=True)
    knn_data = gl.weightmatrix.knnsearch(X,50,similarity='angular',dataset=dataset,metric=metric)

W = gl.weightmatrix.knn(None,k,knn_data=knn_data)
for j in range(np.max(all_train)):

    #Get training data
    train_mask = all_train <= j
    train_idx = index[train_mask]

    #Apply Graph Learning and compute accuracy on testing data
    laplace_labels = gl.ssl.laplace(W).fit_predict(train_idx,labels[train_idx])
    laplace_acc = 100*np.mean(labels[test_mask] == laplace_labels[test_mask])

    print('%d,%.2f'%(len(train_idx),laplace_acc))
    f_cnnvae_laplace.write('%d,%.2f\n'%(len(train_idx),laplace_acc))

f_cnnvae_laplace.close()
















