import numpy as np
import graphlearning as gl
from scipy import sparse
import scipy.sparse as sps
from scipy.special import softmax

def acquisition_function(C_a, V, candidate_inds, u, method='vopt', plot=False, gamma=0.1):
  assert method in ['uncertainty','vopt','mc','mcvopt']

  #calculate uncertainty terms
  num_classes = u.shape[1]
  u_probs = softmax(u[candidate_inds], axis=1)
  one_hot_predicted_labels = np.eye(num_classes)[np.argmax(u[candidate_inds], axis=1)]
  unc_terms = np.linalg.norm((u_probs - one_hot_predicted_labels), axis=1)

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

def active_learning_loop(train_ind, labels, num_iter, C_a, method, gamma=0.1):
  assert method in ['random','uncertainty','vopt','mc','mcvopt']
  accuracy = np.array([])

  for i in range(num_iter+1):
    if i>0:
      unlabeled_ind = np.delete(np.arange(len(labels)), train_ind)
      if method is 'random': 
        train_ind = np.append(train_ind, np.random.choice(unlabeled_ind))
      else:
        obj_vals = acquisition_function(C_a, v, unlabeled_ind, u, method, gamma=gamma)
        new_train_ind = unlabeled_ind[np.argmax(obj_vals)]
        C_a = update_C_a(C_a, v, [new_train_ind], gamma=gamma)
        train_ind = np.append(train_ind, new_train_ind)

    u = gl.graph_ssl(W, train_ind, labels[train_ind], algorithm='laplace', return_vector=True)
    laplace_labels = np.argmax(u, axis=1)
    laplace_acc = gl.accuracy(labels, laplace_labels, len(train_ind))
    print('Laplace: %d labels, %.3f'%(len(train_ind),laplace_acc))
    accuracy = np.append(accuracy, laplace_acc)

  return train_ind, accuracy

def toy_dataset():
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

  #Return a knn weight matrix 
  return gl.knn_weight_matrix(15, X), labels

if __name__ == "__main__":
  # Construct weight matrix and labels of the dataset
  W, labels = toy_dataset()

  # Calculate eigenvalues and eigenvectors
  L = sps.csgraph.laplacian(W, normed=False)
  evals, evecs = sparse.linalg.eigsh(L, k=200, which='SM')
  evals, evecs = evals.real, evecs.real
  d, v = (0.00025 + evals[:]), evecs[:,:]  # we will ignore the first eigenvalue/vector

  # Set initial labeled set
  np.random.seed(2)
  train_ind = gl.randomize_labels(labels, 1)
  unlabeled_ind = np.delete(np.arange(W.shape[0]), train_ind)
  np.random.seed()

  # Run Active Learning Test
  gamma = 0.1
  C_a = np.linalg.inv(np.diag(d) + v[train_ind,:].T @ v[train_ind,:] / gamma**2.) # M by M covariance matrix
  mcvopt_train, mcvopt_accuracy = active_learning_loop(train_ind, labels, 10, C_a, 'mcvopt', gamma=gamma)

