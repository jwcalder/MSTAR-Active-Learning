# MSTAR-Active-Learning

Python package for MSTAR data. To clone the package, run the command

```
git clone git@github.com:jwcalder/MSTAR-Active-Learning.git
```

To load the data in Python and display some images, run the code below.

```
import utils
import numpy as np
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
plt.show()
```

# Run Active Learning
To run active learning tests on the MSTAR data, you can simply run the following code in the command line (terminal):
```
python Python/mstar_run_al.py 
```
* A variety of flags are available for customizing this process; view them all via the ``--help`` flag in the above script call 

### Data Pipeline Description and Active Learning
For the sake of completeness, we briefly describe our overall data pipeline that feeds into our active learning setup:
1. Train Variational Autoencoder (VAE) on the set of input SAR images (i.e. trained __without use of labeled data__)
2. Construct similarity graph weight matrix ``W`` from extracted VAE embeddings
3. Define labeled set of points, ``train_ind`` 
4. Run graph-based SSL via the methods of [GraphLearning package](https://github.com/jwcalder/GraphLearning.git)

Active learning then improves the performance of the chosen graph-based SSL method by selecting informative points in iterative feedback loop, called the "Active Learning Process". This process alternates between (1) computing graph-based SSL classifier given __current__ labeled data and (2) selecting new points to "hand-label" via the use of an ``acquistion function``. 

For the acquisition functions used herein, we use a proxy graph-based model that uses the geometric information contained in the first ``M`` eigenvalues and eigenvectors of the positive semi-definite graph Laplacian matrix used in graph-based methods. The properties of this proxy model are more efficient to compute to allow us to make faster decisions of points to label in the active learning process. The acquisition functions implemented herein are:
* ``uncertainty``: Uncertainty sampling, with a variety of uncertainty measures implemented (``largest_margin``, ``norm``, ``least_confidence``, ``entropy``, ``smallest_margin``)
* ``vopt``: VOpt
* ``mc``: Model Change
* ``mcvopt``: Novel combination of Model Change and VOpt acquisition functions


# CNN and VAE Representation Tests
We ran some initial tests to gauge whether or not graph-based learning using unsupervised VAE embeddings would be competitive. Can recreate the following plot by running the following:
```
# the code for doing this test
```
