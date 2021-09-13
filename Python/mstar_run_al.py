import numpy as np
import graphlearning as gl
from scipy import sparse
import scipy.sparse as sps
from scipy.special import softmax
import os
from argparse import ArgumentParser
from tqdm import tqdm

import torch
import utils
import models
from active_learning import *

DEFAULT_CNN = "SAR10_CNN_2753"
METHODS = ['random', 'uncertainty', 'mc', 'mcvopt', 'vopt']

if __name__ == "__main__":
    parser = ArgumentParser(description='Run active learning test on MSTAR dataset.')
    parser.add_argument("--iters", type=int, default=10, help="number of active learning iterations")
    parser.add_argument("--M", type=int, default=50, help="number of eigenvalues to use in truncation")
    parser.add_argument("--data", type=str, default="SAR10_cnn.npz", help="filepath to .npz file that contains data, labels.")
    parser.add_argument("--gamma", type=float, default=0.5, help="gamma constant for Gaussian Regression covariance calculations")
    parser.add_argument("--seed", type=int, default=2, help="random number generator seed for train_ind chocies")
    args = parser.parse_args()


    # load in CNN representations, build graph
    print("--------- Load in Data and Graph Construction --------")
    if not os.path.exists(args.data):
        print(f"The file {args.data} does not exist, creating representations from {DEFAULT_CNN}")
        #Load data and stack mag,real phase, and imaginary phase together
        hdr, fields, mag, phase = utils.load_MSTAR()
        data = utils.polar_transform(mag, phase)
        data = torch.from_numpy(data).float()

        #Convert target names to integer labels
        labels, target_names = utils.targets_to_labels(hdr)

        # Instantiate CNN model
        model = models.CNN()
        model = torch.load(f"models/{DEFAULT_CNN}.pt", map_location=torch.device('cpu'))
        # train_idx = np.load(f"models/{DEFAULT_CNN}_training_indices.npy")
        # test_idx = np.delete(np.arange(len(labels)), train_idx)

        out = model.encode(data[:5,:,:,:])

        # feed data through
        print("\tPushing data through CNN")
        batch_size = 500
        i = 0
        N, d = data.shape[0], out.shape[-1]
        X = np.empty((N, d))
        for it in tqdm(range(data.shape[0]//batch_size + 1), total=data.shape[0]//batch_size+1):
            ofs = min(batch_size, N-i)
            X[i:i+ofs] = model.encode(data[i:i+ofs]).detach().numpy()
            i += ofs

        print(f"\tSaving representations to {args.data}...")
        np.savez(f'{args.data}', data=X, labels=labels)
    else:
        mstar = np.load(args.data)
        X, labels = mstar['data'], mstar['labels']

    assert X.shape[0] == labels.shape[0]
    print("\tConstructing Graph...")
    W = gl.knn_weight_matrix(20, X)

    # Calculate eigenvalues and eigenvectors
    print("\tCalculating Eigenvalues/Eigenvectors")
    L = sps.csgraph.laplacian(W, normed=False)
    evals, evecs = sparse.linalg.eigsh(L, k=args.M, which='SM')
    evals, evecs = evals.real, evecs.real
    d, v = evals[1:], evecs[:,1:]  # we will ignore the first eigenvalue/vector

    print("-------- Run Active Learning --------")
    print(f"\tAcquisition Functions = {METHODS}")
    print(f"\titerations = {args.iters}, # evals = {args.M}, gamma = {args.gamma}")
    saveloc = os.path.join("results", f"{args.iters}-{args.M}-{args.gamma}")
    if not os.path.exists(saveloc):
        os.makedirs(saveloc)
    print(f"\tsaving results to results/{args.iters}-{args.M}-{args.gamma}/")
    print()



    for acq in METHODS:
        print(f"Acquisition Function = {acq}")

        # Set initial labeled set
        np.random.seed(args.seed)
        train_ind = gl.randomize_labels(labels, 1)
        unlabeled_ind = np.delete(np.arange(W.shape[0]), train_ind)

        # Run Active Learning Test
        train_ind, accuracy = active_learning_loop(W, d, v, train_ind, labels, args.iters, acq, gamma=args.gamma)

        np.savez(os.path.join(saveloc, f"{acq}.npz"), train_ind=train_ind, accuracy=accuracy)
        print("\n")
