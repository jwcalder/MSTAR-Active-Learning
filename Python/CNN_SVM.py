#Code for running graph learning on features from
#pretrained CNN models

import numpy as np
import sys
import os
import graphlearning as gl

#Imports from our package
import utils

#Load MSTAR and CNN models
hdr, fields, mag, phase = utils.load_MSTAR()
cnn_models, cnn_train_idx, cnn_num_train =  utils.get_cnn_models()
all_train = np.load('../models/SAR10_CNN_all_train.npy')
index = np.arange(len(all_train))

#Get labels and corresponding target names
_,test_mask,_ = utils.train_test_split(hdr,1)
labels, target_names = utils.targets_to_labels(hdr)

#Open results file to write accuracy
f_SVM = open('../results/SAR10_CNN_SVM_accuracy.csv',"w")
f_SVM.write('Number of CNN Labels,Number of Labels,Accuracy\n')
print('Number of CNN Labels,Number of Labels,Accuracy')

#Loop over CNN models and apply graph learning
for i, model, fname_train_idx, num_train in zip(list(range(len(cnn_num_train))),cnn_models, cnn_train_idx, cnn_num_train):

    #Load training indices (for training the CNN)
    cnn_train_idx = np.load(fname_train_idx)
   
    #Check if knn data is saved, to save time
    X = utils.encodeMSTAR(model, use_phase=True)

    #Increase the labeled data from the CNN training set up to the full training set
    for j in range(i,len(cnn_num_train)):

        #Get training data
        train_mask = all_train <= j
        train_idx = index[train_mask]

        #Apply Graph Learning and compute accuracy on testing data
        laplace_labels = gl.graph_ssl(W,train_idx,labels[train_idx],algorithm='laplace')
        laplace_acc = 100*np.mean(labels[test_mask] == laplace_labels[test_mask])

        print('%d,%d,%.2f'%(num_train,len(train_idx),laplace_acc))
        f_SVM.write('%d,%d,%.2f\n'%(num_train,len(train_idx),laplace_acc))
