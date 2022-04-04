# MSTAR-Active-Learning

Ths is a Python package for running graph-based active learning algorithms on the [MSTAR](https://www.sdms.afrl.af.mil/index.php?collection=mstar) dataset. This package reproduces experiments from the paper

K. Miller, X. Baca, J. Mauro, J. Setiadi, Z. Shi, J. Calder, and A. Bertozzi. [Graph-based active learning for semi-supervised classification of SAR data.](https://arxiv.org/abs/2204.00005), To appear in SPIE Defense and Commercial Sensing: Algorithms for Synthetic Aperture Radar Imagery XXIX, 2022.

To clone the package, run the command
```
git clone git@github.com:jwcalder/MSTAR-Active-Learning.git
```
To install dependency packages run
```
pip install -r requirements.txt
```

To load the data in Python and display some images, run the code below.

```
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
```

# Run Active Learning
To run active learning tests on the MSTAR data, you can simply run the following code in the command line (terminal):
```
cd Python
python mstar_run_al.py --iters 200
```
* A variety of flags are available for customizing this process; view them all via the ``--help`` flag in the above script call

## NGA-specific test
The test requested by NGA is prepared with the correct variables in the script ``mstar_run_al_nga.py``.
  * Has all the same flags and variables for adjusting this test, similar to main script ,``mstar_run_al.py``

### Data Pipeline Description and Active Learning
For the sake of completeness, we briefly describe our overall data pipeline that feeds into our active learning setup:
1. Train Variational Autoencoder (VAE) on the set of input SAR images (i.e. trained __without use of labeled data__)
2. Construct similarity graph weight matrix ``W`` from extracted VAE embeddings
3. Define labeled set of points, ``train_ind``
4. Run graph-based SSL via the methods of [GraphLearning package](https://github.com/jwcalder/GraphLearning.git)

Active learning then improves the performance of the chosen graph-based SSL method by selecting informative points in iterative feedback loop, called the "Active Learning Process". This process alternates between (1) computing graph-based SSL classifier given __current__ labeled data and (2) selecting new points to "hand-label" via the use of an **acquistion function**.

For the acquisition functions used herein, we use a proxy graph-based model that uses the geometric information contained in the first ``M`` eigenvalues and eigenvectors of the positive semi-definite graph Laplacian matrix used in graph-based methods. The properties of this proxy model are more efficient to compute to allow us to make faster decisions of points to label in the active learning process. The acquisition functions implemented herein are:
* ``uncertainty``: Uncertainty sampling, with a variety of uncertainty measures implemented (``largest_margin``, ``norm``, ``least_confidence``, ``entropy``, ``smallest_margin``)
* ``vopt``: VOpt, variance minimization adaptation of [Ji and Han (2012)](https://proceedings.mlr.press/v22/ji12.html)
* ``mc``: Model Change of [Miller and Bertozzi (2021)](https://arxiv.org/abs/2110.07739)
* ``mcvopt``: Novel combination of Model Change and VOpt acquisition functions

__Note: Can change which acquisition functions are run in the test by changing the variable ``METHODS`` hard-coded into the ``mstar_run_al.py`` script located on line 25.__


# CNN and VAE Representation Tests
We ran some initial tests to gauge whether or not graph-based learning using unsupervised VAE embeddings would be competitive, and [this plot](figures/CNN_Laplace.pdf) summarizes our result. All results are currently saved in the ``models`` and ``figures`` directories. However, one can recreate the training and testing process that created the plot by running the following scripts:
```
cd Python
python trainVAE.py          # train the VAE on all the unlabeled data and save the learned model
python trainCNN.py          # train CNNs with 5%, 10%, 15%, etc of the training data and save the learned models
python CNN_ML.py            # apply a variety of ML methods on each of the learned CNN representations
python CNN_graphlearning.py # apply graph learning on the CNN and VAE representations
python generate_figures.py  # generate plot of accuracies of various ML models on the different CNN and VAE representations
```

# Preprocessing
The MSTAR raw data has already been preprocessed to extract the magnitude and phase images, crop to common sizes, and extract the header information from each image. To re-run any of this preprocessing, run the script
```
python MSTARpreprocess.py
```
To use the preprocessing script, you need to download the raw MSTAR .zip files and place them in an approproate directory for the script to access. Please see the header of the script for details.
