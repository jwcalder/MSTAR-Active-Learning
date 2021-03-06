#Code for running ML on features from
#pretrained CNN models

import numpy as np
import sys
import os
import graphlearning as gl
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

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
f_RF = open('../results/SAR10_CNN_RF_accuracy.csv',"w")
f_RF.write('Number of CNN Labels,Number of Labels,Accuracy\n')
f_NN = open('../results/SAR10_CNN_NN_accuracy.csv',"w")
f_NN.write('Number of CNN Labels,Number of Labels,Accuracy\n')

print('Classifier,Number of CNN Labels,Number of Labels,Accuracy')

#Loop over CNN models and apply graph learning
for i, model, fname_train_idx, num_train in zip(list(range(len(cnn_num_train))),cnn_models, cnn_train_idx, cnn_num_train):

    #Load training indices (for training the CNN)
    cnn_train_idx = np.load(fname_train_idx)
   
    #Encode MSTAR data with CNN model
    X = utils.encodeMSTAR(model, use_phase=True)

    #Increase the labeled data from the CNN training set up to the full training set
    for j in range(i,len(cnn_num_train)):

        #Get training data
        train_mask = all_train <= j

        #Train SVM
        clfsvm = svm.SVC(kernel='linear')
        clfsvm.fit(X[train_mask,:],labels[train_mask])
        labels_svm = clfsvm.predict(X[test_mask,:])
        acc_svm = 100*np.mean(labels[test_mask] == labels_svm)
        print('SVM,%d,%d,%.2f'%(num_train,np.sum(train_mask),acc_svm))
        f_SVM.write('%d,%d,%.2f\n'%(num_train,np.sum(train_mask),acc_svm))
        
        #Train random forests
        clfrf = RandomForestClassifier()
        clfrf.fit(X[train_mask,:],labels[train_mask])
        labels_rf = clfrf.predict(X[test_mask,:])
        acc_rf = 100*np.mean(labels[test_mask] == labels_rf)
        print('Random Forests,%d,%d,%.2f'%(num_train,np.sum(train_mask),acc_rf))
        f_RF.write('%d,%d,%.2f\n'%(num_train,np.sum(train_mask),acc_rf))

        #Nearest neighbor classifier
        clfNN = KNeighborsClassifier(n_neighbors=5)
        clfNN.fit(X[train_mask,:],labels[train_mask])
        labels_NN = clfNN.predict(X[test_mask,:])
        acc_NN = 100*np.mean(labels[test_mask] == labels_NN)
        print('NN,%d,%d,%.2f'%(num_train,np.sum(train_mask),acc_NN))
        f_NN.write('%d,%d,%.2f\n'%(num_train,np.sum(train_mask),acc_NN))


f_NN.close()
f_RF.close()
f_NN.close()







