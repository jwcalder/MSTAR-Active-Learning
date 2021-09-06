# MSTAR-Active-Learning

Python package for MSTAR data. To load the data in Python and display some images

```
import utils
import matplotlib.pyplot as plt

hdr,fields,mag,phase = utils.load_MSTAR()

numx,numy = 3,3
fig, axs = plt.subplots(numx,numy,figsize=(15,15))
n = mag.shape[0]
R = np.random.permutation(n)
for i in range(numx):
    for j in range(numy):
        img = mag[R[i],:,:]
        axs[i,j].imshow(img,cmap='gray')
        axs[i,j].set_title(hdr[R[i],0])
```
