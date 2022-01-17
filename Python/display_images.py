import utils
import numpy as np
import matplotlib.pyplot as plt

hdr,fields,mag,phase = utils.load_MSTAR()
n = mag.shape[0]
R = np.random.permutation(n)

numx,numy = 3,3
fig, axs = plt.subplots(numx,numy,figsize=(15,15))
fig.suptitle('Magnitude Images')
for i in range(numx):
    for j in range(numy):
        img = mag[R[i],:,:]
        axs[i,j].imshow(img,cmap='gray')
        axs[i,j].set_title(hdr[R[i],0])

fig, axs = plt.subplots(numx,numy,figsize=(15,15))
fig.suptitle('Phase Images')
for i in range(numx):
    for j in range(numy):
        img = phase[R[i],:,:]
        axs[i,j].imshow(img,cmap='gray')
        axs[i,j].set_title(hdr[R[i],0])

plt.show()
