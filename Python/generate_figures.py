import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib
import pandas as pd
import os
import numpy as np

#Our package imports
import utils

#General plot settings
legend_fontsize = 12
label_fontsize = 16
fontsize = 16
matplotlib.rcParams.update({'font.size': fontsize})
styles = ['^b-','or-','dg-','sk-','pm-','xc-','*y-']
#styles = ['b-','r-','g-','k-','m-','c-','y-']

#CNN vs CNN+Laplace
ax = plt.figure().gca()

#CNNVAE Laplace learning
df = pd.read_csv('../results/SAR10_CNNVAE_laplace_accuracy.csv')
num_labels = df['Number of Labels'].values
laplace_accuracy = df['Accuracy'].values
ax.plot(num_labels,laplace_accuracy,styles[0],label='CNNVAE & Laplace')

#CNN Laplace learning
df = pd.read_csv('../results/SAR10_CNN_laplace_accuracy.csv')
num_labels = df['Number of Labels'].values
num_cnn_labels = df['Number of CNN Labels'].values
I = num_labels == num_cnn_labels
laplace_accuracy = df['Accuracy'].values
laplace_accuracy = laplace_accuracy[I]
num_labels = num_labels[I]
ax.plot(num_labels,laplace_accuracy,styles[1],label='CNN & Laplace')

#CNN and ML
#df_NN = pd.read_csv('../results/SAR10_CNN_NN_accuracy.csv')
#df_RF = pd.read_csv('../results/SAR10_CNN_RF_accuracy.csv')
#df_SVM = pd.read_csv('../results/SAR10_CNN_SVM_accuracy.csv')
#ax.plot(num_labels,df_NN['Accuracy'].values[I],styles[2],label='CNN & NN')
#ax.plot(num_labels,df_RF['Accuracy'].values[I],styles[3],label='CNN & RF')
#ax.plot(num_labels,df_SVM['Accuracy'].values[I],styles[4],label='CNN & SVM')
#cnn_models, cnn_train_idx, cnn_num_train =  utils.get_cnn_models()
#cnn_accuracy = np.zeros_like(laplace_accuracy)
#max_cnn_accuracy = np.zeros_like(laplace_accuracy)
#for i in range(len(cnn_models)):
#    df = pd.read_csv('../results/SAR10_CNN_%d_accuracy.csv'%cnn_num_train[i])
#    acc = df['Test Accuracy'].values
#    cnn_accuracy[i] = acc[-1]
#    #max_cnn_accuracy[i] = np.max(acc)
#ax.plot(num_labels,cnn_accuracy,styles[5],label='CNN')
#ax.plot(num_labels,max_cnn_accuracy,styles[2],label='CNNmax')

#Labels and legends
plt.xlabel('Number of labels',fontsize=label_fontsize)
plt.ylabel('Accuracy (%)',fontsize=label_fontsize)
plt.legend(loc='lower right',fontsize=legend_fontsize)
#plt.title('CNN & Laplace Learning')
plt.tight_layout()
plt.grid(True)
#plt.ylim(ylim)
#ax.xaxis.set_major_locator(MaxNLocator(integer=True))

#Save figures
plt.savefig('../figures/CNN_Laplace.eps')
plt.savefig('../figures/CNN_Laplace.pdf')
