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

#Get labels and corresponding target names
labels, target_names = utils.targets_to_labels(hdr)

#Open results file to write accuracy
f_laplace = open('../results/SAR10_CNN_laplace_accuracy.csv',"w")
f_laplace.write('Number of Labels,Accuracy\n')
print('Number of Labels,Accuracy')

#Loop over CNN models and apply graph learning
for model, fname_train_idx, num_train in zip(cnn_models, cnn_train_idx, cnn_num_train):

    #Load training indices (for training the CNN)
    train_idx = np.load(fname_train_idx)

    #Check if knn data is saved, to save time
    #Dataset and metric name
    dataset = 'SAR10'
    metric = model[16:-3]
    try:
        I,J,D = gl.load_kNN_data(dataset,metric=metric)
    except:
        X = utils.encodeMSTAR(model, use_phase=True)
        I,J,D = gl.knnsearch_annoy(X,50,similarity='angular',dataset=dataset,metric=metric)

    #Build weight matrix
    W = gl.weight_matrix(I,J,D,k)

    #Apply Graph Learning
    laplace_labels = gl.graph_ssl(W,train_idx,labels[train_idx],algorithm='laplace')
    laplace_acc = gl.accuracy(labels,laplace_labels,len(train_idx))

    print('%d,%.2f'%(num_train,laplace_acc))
    f_laplace.write('%d,%.2f\n'%(num_train,laplace_acc))
