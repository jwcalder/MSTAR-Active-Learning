'''
Main Python script for running overall active learning process tests on MSTAR data.
    * Run this script through command line (terminal).
    * View parameter descriptions with "python mstar_run_al.py --help"
'''

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

# Make sure results directory exists to put the active learning results
RESULTSDIR = os.path.join("..", "results", "al_results")
if not os.path.exists(RESULTSDIR):
    os.makedirs(RESULTSDIR)

# Make sure a directory to store eigenvalue and eigenvector data so don't have to recompute with every test
EIGDIR = os.path.join("..", "eigData")
if not os.path.exists(EIGDIR):
    os.makedirs(EIGDIR)

if __name__ == "__main__":
    parser = ArgumentParser(description='Run active learning test on MSTAR dataset.')
    parser.add_argument("--vae_fname", type=str, default="SAR10_CNNVAE", help="string of CNNVAE model name to use for representations, located in ./models directory. Ensure this is a VAE model by having string 'VAE' in the model name.")
    parser.add_argument("--iters", type=int, default=500, help="number of active learning iterations")
    parser.add_argument("--M", type=int, default=200, help="number of eigenvalues to use in truncation")
    parser.add_argument("--knn", type=int, default=20, help="number of knn to use in graph construction")
    parser.add_argument("--gamma", type=float, default=0.5, help="gamma constant for Gaussian Regression covariance calculations")
    parser.add_argument("--seed", type=int, default=2, help="random number generator seed for train_ind choices")
    parser.add_argument("--num_per_class", type=int, default=1, help="number of initially labeled points per class")
    parser.add_argument("--algorithm", type=str, default="laplace", help="Graphlearning graph-based ssl algorithm to use for accuracy calculations")
    parser.add_argument("--plot", type=bool, default=False, help="Set to True to save plot of accuracy results")
    parser.add_argument("--tsne", type=bool, default=False, help="Set to True to visualize t-sne embedding of selected points")
    args = parser.parse_args()

    print("-"*30)
    print(f"MSTAR GBSSL Active Learning Tests - Using {args.vae_fname} Representations")
    print("-"*30)
    print(f"\titers = {args.iters}, num_per_class = {args.num_per_class}")
    print(f"\tknn = {args.knn}, M (num evals) = {args.M}")
    print(f"\talgorithm = {args.algorithm}, seed = {args.seed}")
    print(f"\tplot = {args.plot}")
    print()

    assert "vae" in args.vae_fname.lower() # ensure that we are using VAE computed representations, NOT one of the supervised CNN model's representations

    # Load MSTAR and CNN models
    hdr, fields, mag, phase = utils.load_MSTAR()

    # Get labels and corresponding target names
    train_mask, test_mask, _ = utils.train_test_split(hdr,1)
    labels, target_names = utils.targets_to_labels(hdr)

    # Find specified CNNVAE model's filepath
    model_fpath = os.path.join("..", "models", args.vae_fname + ".pt")
    assert os.path.exists(model_fpath)

    # Define and make results filepath
    results_fpath = args.vae_fname + f"_{args.algorithm}_{args.knn}_{args.M}_{args.gamma}_{args.seed}_{args.num_per_class}_{args.iters}"
    if not os.path.exists(os.path.join(RESULTSDIR, results_fpath)):
        os.makedirs(os.path.join(RESULTSDIR, results_fpath))
    print(f"Saving results to {RESULTSDIR}/{results_fpath}/...")
    print("\t filename format: {vae_fname}_{algorithm}_{knn}_{M}_{gamma}_{seed}_{num_per_class}_{iters}/")
    print()


    # Define dataset name and vae "metric" identifier as well as training set indicies
    dataset, metric = args.vae_fname.split("_")
    train_idx_all = np.where(train_mask)[0]

    # Graph Construction -- check if previously computed
    try:
        I,J,D = gl.load_kNN_data(dataset, metric=metric)
    except:
        X = utils.encodeMSTAR(model_fpath, use_phase=True)
        I,J,D = gl.knnsearch_annoy(X,50,similarity='angular',dataset=dataset,metric=metric)

    W = gl.weight_matrix(I,J,D,args.knn)
    N = W.shape[0]

    # Calculate (or load in previously computed) eigenvalues and eigenvectors of
    eig_fpath = os.path.join(EIGDIR, f"{args.vae_fname}_{args.knn}_{args.M}.npz")
    if not os.path.exists(eig_fpath):
        # Calculate eigenvalues and eigenvectors of unnormalized graph Laplacian if not previously calculated
        print("Calculating Eigenvalues/Eigenvectors...")
        L = sps.csgraph.laplacian(W, normed=False)
        evals, evecs = sparse.linalg.eigsh(L, k=args.M+1, which='SM')
        evals, evecs = evals.real, evecs.real
        evals, evecs = evals[1:], evecs[:,1:]  # we will ignore the first eigenvalue/vector



        # Also compute normalized graph laplacian eigenvectors for use in some GraphLearning graph_ssl functions (e.g. "mbo")
        n = W.shape[0]
        deg = gl.degrees(W)
        m = np.sum(deg)/2
        gamma = 0
        Lnorm = gl.graph_laplacian(W,norm="normalized")
        def Mnorm(v):
            v = v.flatten()
            return (Lnorm*v).flatten() + (gamma/m)*(deg.T@v)*deg
        Anorm = sparse.linalg.LinearOperator((n,n), matvec=Mnorm)
        vals_norm, vecs_norm = sparse.linalg.eigs(Anorm,k=300,which='SM')
        vals_norm = vals_norm.real; vecs_norm = vecs_norm.real

        print(f"\tSaved to {eig_fpath}")
        np.savez(eig_fpath, evals=evals, evecs=evecs, vals_norm=vals_norm, vecs_norm=vecs_norm)
    else:
        print(f"Found saved eigendata at {eig_fpath}")
        eigdata = np.load(eig_fpath)
        evals, evecs, vals_norm, vecs_norm = eigdata["evals"], eigdata["evecs"], eigdata["vals_norm"], eigdata["vecs_norm"]


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


        # Save initially labeled set
        if not os.path.exists(os.path.join(RESULTSDIR, results_fpath, "init_labeled.npy")):
            np.save(os.path.join(RESULTSDIR, results_fpath, "init_labeled.npy"), train_ind)


        # Run Active Learning Test for this current acqusition function
        train_ind, accuracy = active_learning_loop(W, evals, evecs, train_ind, labels, args.iters, acq, train_idx_all=train_idx_all, test_mask=test_mask, gamma=args.gamma, algorithm=args.algorithm, vals_norm = vals_norm, vecs_norm = vecs_norm)

        results_df[acq+"_choices"] = np.concatenate(([-1], train_ind[-args.iters:]))
        results_df[acq+"_acc"] = accuracy

        print("\n")

    results_df.to_csv(os.path.join(RESULTSDIR, results_fpath, "results.csv"))
    print(f"Results saved in directory {os.path.join(RESULTSDIR, results_fpath)}/")

    # Creates t-SNE visualizations of dataset, train/test split, and queried active learning points
    if args.tsne:
        from sklearn.manifold import TSNE
        tsne_data_path = os.path.join("..", "results", f"tsne_{args.vae_fname}.npy")
        if not os.path.exists(tsne_data_path):
            X = utils.encodeMSTAR(model_fpath, use_phase=True)
            tsne_embedded_data = TSNE(n_components=2, init='pca', learning_rate='auto').fit_transform(X)
            np.save(tsne_data_path, tsne_embedded_data)
            print(f"tSNE embedding data saved to {tsne_data_path}")
        else:
            print(f"Found saved t-SNE embedding at {tsne_data_path}")
            tsne_embedded_data = np.load(tsne_data_path)

        # Plot the t-SNE embedding of MSTAR, if not already exist
        if not os.path.exists(os.path.join("..", "results", f"tsne_{args.vae_fname}.png")):
            plt.figure()
            plt.scatter(tsne_embedded_data[:,0], tsne_embedded_data[:,1], c=labels, s=.5)
            plt.title("t-SNE Embedding of MSTAR Data")
            plt.savefig(os.path.join("..", "results", f"tsne_{args.vae_fname}.png"))

            # Visualize the train/test split with the t-SNE Embedding
            plt.figure()
            plt.scatter(tsne_embedded_data[train_idx_all,0], tsne_embedded_data[train_idx_all,1], c = 'blue', label = "Train points", s=.5)
            plt.scatter(tsne_embedded_data[test_mask,0], tsne_embedded_data[test_mask,1], c = 'red', label = "Test points", s=.5)
            plt.title("t-SNE Embedding of Train Test Split")
            plt.legend()
            plt.savefig(os.path.join("..", "results", f"tsne_{args.vae_fname}_train_test.png"))


        # Visualize the points queried by each active learning method for this test
        for method in METHODS:
            plt.figure()
            indexes_queried = results_df[method + "_choices"]
            plt.scatter(tsne_embedded_data[:,0], tsne_embedded_data[:,1], c = labels, s=.5)
            plt.scatter(tsne_embedded_data[indexes_queried, 0], tsne_embedded_data[indexes_queried, 1], c = 'red', marker = '*', label = "Active learning points")
            plt.title(f"Query Points from {method}")
            plt.legend()
            plt.savefig(os.path.join(RESULTSDIR, results_fpath, "tsne_" + method + "_query_points.png"))


    # Plots the accuracies of the tested active learning methods and saves to results folder
    if args.plot:

        plt.figure()

        x = np.arange(args.num_per_class, args.num_per_class + args.iters + 1)

        # General plot settings
        legend_fontsize = 12
        label_fontsize = 16
        fontsize = 16
        # matplotlib.rcParams.update({'font.size': fontsize})
        styles = ['^b-','or-','dg-','pm-','xc-','sk-', '*y-']

        skip = 12

        for i, method in enumerate(METHODS):
            plt.plot(x[::skip], 100*results_df[method + "_acc"][::skip], styles[i], label = method + " accuracy")

        plt.xlabel("Number of Labeled Points")
        plt.ylabel("Accuracy %")
        plt.title("Active Learning on " + args.vae_fname)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        text =  "iters = " + str(args.iters) + ", num_per_class = " + str(args.num_per_class) + ", knn = " + str(args.knn) + ", gamma = " + str(args.gamma) + ", M (num evals) = " + str(args.M) + ", algorithm = " + str(args.algorithm) + ", seed = " + str(args.seed)
        #plt.figtext(.5, .99, text, wrap= True, horizontalalignment = 'center', fontsize=6) #puts description at top of plot of which parameters were used to help with reproducibility


        plt.savefig(os.path.join(RESULTSDIR, results_fpath, "results.png"))
