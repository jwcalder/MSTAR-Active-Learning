import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib
import pandas as pd
import os
import numpy as np

#Our package imports
import utils

#General plot settings
legend_fontsize = 16
label_fontsize = 16
fontsize = 16
matplotlib.rcParams.update({'font.size': fontsize})
styles = ['^b-','or-','dg-','sk-','pm-','xc-','*y-']

#CNN vs CNN+Laplace
ax = plt.figure().gca()

#Laplace learning
df = pd.read_csv('../results/SAR10_CNN_laplace_accuracy.csv')
num_labels = df['Number of Labels'].values
laplace_accuracy = df['Accuracy'].values
ax.plot(num_labels,laplace_accuracy,styles[0],label='CNN & Laplace')

#CNN
cnn_models, cnn_train_idx, cnn_num_train =  utils.get_cnn_models()
cnn_accuracy = np.zeros_like(laplace_accuracy)
max_cnn_accuracy = np.zeros_like(laplace_accuracy)
for i in range(len(cnn_models)):
    df = pd.read_csv('../results/SAR10_CNN_%d_accuracy.csv'%cnn_num_train[i])
    acc = df['Test Accuracy'].values
    cnn_accuracy[i] = acc[-1]
    max_cnn_accuracy[i] = np.max(acc)
ax.plot(num_labels,cnn_accuracy,styles[1],label='CNN')
ax.plot(num_labels,max_cnn_accuracy,styles[2],label='CNNmax')

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
