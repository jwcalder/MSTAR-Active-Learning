import numpy as np
import graphlearning as gl
from scipy import sparse
import scipy.sparse as sps
from scipy.special import softmax
from argparse import ArgumentParser

def acquisition_function(C_a, V, candidate_inds, u, method='vopt', uncertainty_method = 'norm', plot=False, gamma=0.1):
    assert method in ['uncertainty','vopt','mc','mcvopt']

    #calculate uncertainty terms
    num_classes = u.shape[1]
    u_probs = softmax(u[candidate_inds], axis=1)
    one_hot_predicted_labels = np.eye(num_classes)[np.argmax(u[candidate_inds], axis=1)]

    uncertainty_methods = { #Dictionary to map uncertainty method to the uncertainty terms used in that method
    "norm": np.linalg.norm((u_probs - one_hot_predicted_labels), axis=1),
    "entropy": np.max(u[candidate_inds], axis=1) - np.sum(u[candidate_inds]*np.log(u[candidate_inds]+.00001), axis=1),
    "least_confidence": np.ones((u[candidate_inds].shape[0],)) - np.max(u[candidate_inds], axis=1),
    "smallest_margin": 1-(np.sort(u[candidate_inds])[:,num_classes-1] - np.sort(u[candidate_inds])[:, num_classes-2]),
    "largest_margin": 1-(np.sort(u[candidate_inds])[:,num_classes-1] - np.sort(u[candidate_inds])[:, 0]),
    "random": np.random.random(u[candidate_inds].shape[0]),
    }

    unc_terms = uncertainty_methods.get(uncertainty_method)

    if method == 'uncertainty':
        return unc_terms

    Cavk = C_a @ V[candidate_inds,:].T
    col_norms = np.linalg.norm(Cavk, axis=0)
    diag_terms = (gamma**2. + np.array([np.inner(V[k,:], Cavk[:, i]) for i,k in enumerate(candidate_inds)]))

    if method == 'vopt':
        return col_norms**2. / diag_terms

    if method == 'mc':
        return unc_terms * col_norms / diag_terms

    else:
        return unc_terms * col_norms **2. / diag_terms

def update_C_a(C_a, V, Q, gamma=0.1):
    for k in Q:
        vk = V[k]
        Cavk = C_a @ vk
        ip = np.inner(vk, Cavk)
        C_a -= np.outer(Cavk, Cavk)/(gamma**2. + ip)
    return C_a

def active_learning_loop(W, evals, evecs, train_ind, labels, num_iter, method, all_train_idx=None, test_mask=None, gamma=0.1, algorithm='laplace', uncertainty_method = 'norm', verbose=True):
    assert method in ['random','uncertainty','vopt','mc','mcvopt']
    accuracy = np.array([])
    C_a = np.linalg.inv(np.diag(evals) + evecs[train_ind,:].T @ evecs[train_ind,:] / gamma**2.) # M by M covariance matrix

    for i in range(num_iter+1):
        if i > 0:
            if all_train_idx is None: # assume that all unlabeled indices are potential candidates for active learning
                candidate_inds = np.delete(np.arange(len(labels)), train_ind)
            else:                     # candidate inds must be subset of given "all_train_idx"
                candidate_inds = np.setdiff1d(all_train_idx, train_ind)

            if method == 'random':
                train_ind = np.append(train_ind, np.random.choice(candidate_inds))
            else:
                obj_vals = acquisition_function(C_a, evecs, candidate_inds, u, method, uncertainty_method = uncertainty_method, gamma=gamma)
                new_train_ind = candidate_inds[np.argmax(obj_vals)]
                C_a = update_C_a(C_a, evecs, [new_train_ind], gamma=gamma)
                train_ind = np.append(train_ind, new_train_ind)

        u = gl.graph_ssl(W, train_ind, labels[train_ind], algorithm=algorithm, return_vector=True)
        comp_labels = np.argmax(u, axis=1)
        if test_mask is None:
            comp_acc = gl.accuracy(labels, comp_labels, len(train_ind))
        else:
            comp_acc = np.mean(labels[test_mask] == comp_labels[test_mask])
        if verbose:
            if num_iter >= 50 and i % 10 == 0:
                print(f'\t{algorithm}: {len(train_ind)} labeled points, {comp_acc:.3f}')
            elif num_iter < 50:
                print(f'\t{algorithm}: {len(train_ind)} labeled points, {comp_acc:.3f}')


        accuracy = np.append(accuracy, comp_acc)

    return train_ind, accuracy

def toy_dataset(return_X=False):
    #Load data, labels
    n=100
    np.random.seed(12) #set random seed
    std = 0.5
    data1 = std*np.random.randn(n,2) #upper left
    data2 = std*np.random.randn(n,2) #top left
    data2[:,0] += 2
    data2[:,1] += 2
    data3 = std*np.random.randn(n,2) #top right
    data3[:,0] += 4
    data3[:,1] += 2
    data4 = std*np.random.randn(n,2) #upper right
    data4[:,0] += 6
    data5 = std*np.random.randn(n,2) #lower right
    data5[:,0] += 6
    data5[:,1] -= 2
    data6 = std*np.random.randn(n,2) #bottom right
    data6[:,0] += 4
    data6[:,1] -= 4
    data7 = std*np.random.randn(n,2) #bottom left
    data7[:,0] += 2
    data7[:,1] -= 4
    data8 = std*np.random.randn(n,2) #lower left
    data8[:,1] -= 2
    X = np.vstack([data1,data2,data3,data4,data5,data6,data7,data8])
    labels = np.hstack([np.zeros(n),np.ones(n),np.zeros(n),np.ones(n),np.zeros(n),
                      np.ones(n),np.zeros(n),np.ones(n)]).astype('int')
    if return_X:
        return gl.knn_weight_matrix(15, X), labels, X
    #Return a knn weight matrix
    return gl.knn_weight_matrix(15, X), labels

if __name__ == "__main__":
    parser = ArgumentParser(description='Run active learning test on toy dataset.')
    parser.add_argument("--method", type=str, default='mcvopt', help="acquisition function for test. ['mc', 'mcvopt', 'vopt', 'uncertainty', 'random']")
    parser.add_argument("--iters", type=int, default=10, help="number of active learning iterations")
    parser.add_argument("--plot", type=int, default=0, help='0/1 flag of whether or not to plot/save final plot of choices')
    parser.add_argument("--M", type=int, default=50, help="number of eigenvalues to use in truncation")
    parser.add_argument("--gamma", type=float, default=0.5, help="gamma constant for Gaussian Regression covariance calculations")
    args = parser.parse_args()

    # Toy example of running active learning code
    print(f"---------- Running toy example with {args.method.upper()} -----------")
    # Construct weight matrix and labels of the dataset
    if args.plot:
        W, labels, X = toy_dataset(return_X=True)
    else:
        W, labels = toy_dataset()

    # Calculate eigenvalues and eigenvectors
    L = sps.csgraph.laplacian(W, normed=False)
    evals, evecs = sparse.linalg.eigsh(L, k=args.M, which='SM')
    evals, evecs = evals.real, evecs.real
    d, v = evals[1:], evecs[:,1:]  # we will ignore the first eigenvalue/vector

    # Set initial labeled set
    np.random.seed(2)
    train_ind = gl.randomize_labels(labels, 1)
    unlabeled_ind = np.delete(np.arange(W.shape[0]), train_ind)
    np.random.seed()

    # Run Active Learning Test
    train_ind, accuracy = active_learning_loop(W, d, v, train_ind, labels, args.iters, args.method, gamma=args.gamma)


    if args.plot:
        import matplotlib.pyplot as plt
        u = gl.graph_ssl(W, train_ind, labels[train_ind], algorithm='laplace', return_vector=True)[:,1]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
        ax1.scatter(X[:,0], X[:,1], c=1.*(u >= 0.5))
        ax1.scatter(X[train_ind,0], X[train_ind,1], c='r', marker='^', s=100)
        ax1.set_title(f"Final Classifier, {args.method.upper()}")
        ax2.scatter(X[:,0], X[:,1], c=u)
        ax2.scatter(X[train_ind,0], X[train_ind,1], c='r', marker='^', s=100)
        ax2.set_title(f"Pre-Threshold Classifier, {args.method.upper()}")
        plt.savefig(f"{args.method}-{args.M}-{args.iters}.png")
        plt.close()
