import numpy as np
import graphlearning as gl
from scipy import sparse
import scipy.sparse as sps
from scipy.special import softmax
import os
import sys
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm

import torch
import utils
import models
import matplotlib.pyplot as plt
from active_learning import *


METHODS = ['random', 'uncertainty', 'mc', 'mcvopt', 'vopt']

# Make sure results directory exists
RESULTSDIR = os.path.join("..", "results", "al_results")
if not os.path.exists(RESULTSDIR):
    os.makedirs(RESULTSDIR)

EIGDIR = os.path.join("..", "eigData")
if not os.path.exists(EIGDIR):
    os.makedirs(EIGDIR)

if __name__ == "__main__":
    parser = ArgumentParser(description='Run active learning test on MSTAR dataset.')
    parser.add_argument("--cnn_fname", type=str, default="SAR10_CNNVAE", help="string of CNN model name to use for representations (including CNNVAE), located in ./models directory")
    parser.add_argument("--iters", type=int, default=10, help="number of active learning iterations")
    parser.add_argument("--M", type=int, default=200, help="number of eigenvalues to use in truncation")
    parser.add_argument("--knn", type=int, default=20, help="number of knn to use in graph construction")
    parser.add_argument("--gamma", type=float, default=0.5, help="gamma constant for Gaussian Regression covariance calculations")
    parser.add_argument("--seed", type=int, default=2, help="random number generator seed for train_ind choices")
    parser.add_argument("--num_per_class", type=int, default=1, help="number of initially labeled points per class")
    parser.add_argument("--algorithm", type=str, default="laplace", help="Graphlearning graph-based ssl algorithm to use for accuracy calculations")
    parser.add_argument("--plot", type=bool, default=False, help="Set to True to save plot of results")
    args = parser.parse_args()

    print("-"*30)
    print(f"MSTAR GBSSL Active Learning Tests - Using {args.cnn_fname} Representations")
    print("-"*30)
    print(f"\titers = {args.iters}, num_per_class = {args.num_per_class}")
    print(f"\tknn = {args.knn}, M (num evals) = {args.M}")
    print(f"\talgorithm = {args.algorithm}, seed = {args.seed}")
    print(f"\tplot = {args.plot}")
    print()
    #Load MSTAR and CNN models
    hdr, fields, mag, phase = utils.load_MSTAR()
    all_train = np.load('../models/SAR10_CNN_all_train.npy')
    index = np.arange(len(all_train))

    #Get labels and corresponding target names
    _,test_mask,_ = utils.train_test_split(hdr,1)
    labels, target_names = utils.targets_to_labels(hdr)


    model_fpath = os.path.join("..", "models", args.cnn_fname + ".pt")
    assert os.path.exists(model_fpath)

    results_fpath = args.cnn_fname + f"_{args.algorithm}_{args.knn}_{args.M}_{args.gamma}_{args.seed}_{args.num_per_class}_{args.iters}"
    if not os.path.exists(os.path.join(RESULTSDIR, results_fpath)):
        os.makedirs(os.path.join(RESULTSDIR, results_fpath))
    print(f"Saving results to {RESULTSDIR}/{results_fpath}/...")
    print("\t filename format: {cnn_fname}_{algorithm}_{knn}_{M}_{gamma}_{seed}_{num_per_class}_{iters}/")
    print()

    if "VAE" in args.cnn_fname:
        dataset, metric = args.cnn_fname.split("_")
        train_idx_all = index
    else:
        dataset = args.cnn_fname.split("_")[0]
        metric = args.cnn_fname[16:]
        # Get training data
        train_idx_all = np.load(os.path.join('..', 'models', args.cnn_fname + "_training_indices.npy"))

    # Graph Construction
    try:
        I,J,D = gl.load_kNN_data(dataset,metric=metric)
    except:
        X = utils.encodeMSTAR(model_fpath, use_phase=True)
        I,J,D = gl.knnsearch_annoy(X,50,similarity='angular',dataset=dataset,metric=metric)

    W = gl.weight_matrix(I,J,D,args.knn)
    N = W.shape[0]

    eig_fpath = os.path.join(EIGDIR, f"{args.cnn_fname}_{args.knn}_{args.M}.npz")
    if not os.path.exists(eig_fpath):
        # Calculate eigenvalues and eigenvectors if not previously calculated
        print("Calculating Eigenvalues/Eigenvectors")
        L = sps.csgraph.laplacian(W, normed=False)
        evals, evecs = sparse.linalg.eigsh(L, k=args.M+1, which='SM')
        evals, evecs = evals.real, evecs.real
        evals, evecs = evals[1:], evecs[:,1:]  # we will ignore the first eigenvalue/vector
        print(f"\tSaved to {eig_fpath}")
        np.savez(eig_fpath, evals=evals, evecs=evecs)
    else:
        print(f"Found saved eigendata at {eig_fpath}")
        eigdata = np.load(eig_fpath)
        evals, evecs = eigdata["evals"], eigdata["evecs"]

    print()
    print("-"*30)
    print("\tActive Learning Tests")
    print("-"*30)

    results_df = pd.DataFrame([]) # instantiate pandas dataframe for recording results

    for acq in METHODS:
        print(f"Acquisition Function = {acq.upper()}")

        # Select initial training set -- Should be same for each method
        train_ind = np.array([], dtype=np.int16)

        for c in np.sort(np.unique(labels)):
            c_ind = np.intersect1d(np.where(labels == c)[0], train_idx_all) # ensure the chosen points are in the correct subset of the dataset
            rng = np.random.default_rng(args.seed) # for reproducibility
            train_ind = np.append(train_ind, rng.choice(c_ind, args.num_per_class, replace=False))


        # save initially labeled set
        if not os.path.exists(os.path.join(RESULTSDIR, results_fpath, "init_labeled.npy")):
            np.save(os.path.join(RESULTSDIR, results_fpath, "init_labeled.npy"), train_ind)


        # Run Active Learning Test
        train_ind, accuracy = active_learning_loop(W, evals, evecs, train_ind, labels, args.iters, acq, all_train_idx=train_idx_all, test_mask=test_mask, gamma=args.gamma, algorithm=args.algorithm)

        results_df[acq+"_choices"] = np.concatenate(([-1], train_ind[-args.iters:]))
        results_df[acq+"_acc"] = accuracy

        print("\n")

    results_df.to_csv(os.path.join(RESULTSDIR, results_fpath, "results.csv"))
    print(f"Results saved in directory {os.path.join(RESULTSDIR, results_fpath)}/")


    if (args.plot): #Plots the accuracies of different active learning methods and saves to results folder

        plt.figure()

        x = np.arange(args.num_per_class, args.num_per_class + args.iters + 1)

        for method in METHODS:
            plt.plot(x, results_df[method + "_acc"], label = method + " accuracy")

        plt.xlabel("Number of Labeled Points")
        plt.ylabel("Accuracy")
        plt.title("Active Learning on " + args.cnn_fname)
        plt.legend()



        text =  "iters = " + str(args.iters) + ", num_per_class = " + str(args.num_per_class) + ", knn = " + str(args.knn) + ", gamma = " + str(args.gamma) + ", M (num evals) = " + str(args.M) + ", algorithm = " + str(args.algorithm) + ", seed = " + str(args.seed)
        plt.figtext(.5, .96, text, wrap= True, horizontalalignment = 'center', fontsize=6) #puts description at top of plot of which parameters were used to help with reproducibility


        plt.savefig(os.path.join(RESULTSDIR, results_fpath, "results.png"))
