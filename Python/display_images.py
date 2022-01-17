import utils
import numpy as np
import matplotlib.pyplot as plt

hdr,fields,mag,phase = utils.load_MSTAR()

numx,numy = 3,3
fig1, axs1 = plt.subplots(numx,numy,figsize=(15,15))
fig2, axs2 = plt.subplots(numx,numy,figsize=(15,15))
fig1.suptitle('Magnitude Images')
fig2.suptitle('Phase Images')
n = mag.shape[0]
R = np.random.permutation(n)
for i in range(numx):
    for j in range(numy):
        img = mag[R[i],:,:]
        axs1[i,j].imshow(img,cmap='gray')
        axs1[i,j].set_title(hdr[R[i],0])
        img = phase[R[i],:,:]
        axs2[i,j].imshow(img,cmap='gray')
        axs2[i,j].set_title(hdr[R[i],0])
plt.show()
