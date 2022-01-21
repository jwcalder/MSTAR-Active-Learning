import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib
import pandas as pd
import os
import numpy as np

#Our package imports
import utils

def MSTAR_image(data,i):
    img = np.hstack((data[i,0,:,:]**0.5,data[i,1,:,:],data[i,2,:,:]))
    img[:,87:89]=1
    img[:,175:177]=1
    return img

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
df_NN = pd.read_csv('../results/SAR10_CNN_NN_accuracy.csv')
df_RF = pd.read_csv('../results/SAR10_CNN_RF_accuracy.csv')
df_SVM = pd.read_csv('../results/SAR10_CNN_SVM_accuracy.csv')
ax.plot(num_labels,df_NN['Accuracy'].values[I],styles[2],label='CNN & NN')
#ax.plot(num_labels,df_RF['Accuracy'].values[I],styles[3],label='CNN & RF')
ax.plot(num_labels,df_SVM['Accuracy'].values[I],styles[4],label='CNN & SVM')
cnn_models, cnn_train_idx, cnn_num_train =  utils.get_cnn_models()
cnn_accuracy = np.zeros_like(laplace_accuracy)
max_cnn_accuracy = np.zeros_like(laplace_accuracy)
for i in range(len(cnn_models)):
    df = pd.read_csv('../results/SAR10_CNN_%d_accuracy.csv'%cnn_num_train[i])
    acc = df['Test Accuracy'].values
    cnn_accuracy[i] = acc[-1]
    #max_cnn_accuracy[i] = np.max(acc)
ax.plot(num_labels,cnn_accuracy,styles[5],label='CNN')
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

#CNN Training Plots
plt.figure()
df = pd.read_csv('../results/SAR10_CNN_3671_accuracy.csv')
epoch = df['Epoch'].values
test_acc = df['Test Accuracy'].values
train_acc = df['Train Accuracy'].values
plt.plot(epoch,test_acc,'b-',label='Test Acc (100% of training data)')
plt.plot(epoch,train_acc,'b--',label='Train Acc (100% of training data)')

df = pd.read_csv('../results/SAR10_CNN_367_accuracy.csv')
epoch = df['Epoch'].values
test_acc = df['Test Accuracy'].values
train_acc = df['Train Accuracy'].values
plt.plot(epoch,test_acc,'r-',label='Test Acc (10% of training data))')
plt.plot(epoch,train_acc,'r--',label='Train Acc (10% of training data)')

plt.xlabel('Epoch',fontsize=label_fontsize)
plt.ylabel('Accuracy (%)',fontsize=label_fontsize)
plt.legend(loc='lower right',fontsize=legend_fontsize)
plt.tight_layout()
plt.grid(True)

#Save figures
plt.savefig('../figures/CNN_train.eps')
plt.savefig('../figures/CNN_train.pdf')



#Generic images
hdr,fields,mag,phase = utils.load_MSTAR()
data = utils.polar_transform(mag,phase)
for i in range(10):
    plt.imsave('../figures/MSTAR_image%d.png'%i, MSTAR_image(data,i), cmap='gray')









