#These are common utilities that are used by many different scripts

import numpy as np
import torch
import os
import glob

#Our package imports
import models


def get_cnn_models(model_dir = 'models'):
    '''Returns a list of the CNN model names and number of training points

    Parameters
    ----------
    model_dir : (optional, default = 'models')

    Returns
    -------
    cnn_models : A python list of the strings containing all CNN models
                 in model_dir ending in *.pt
    cnn_train_idx : A pytyhon list of .npy files giving training points for each model
    cnn_num_train : A python list giving the number of training points for each model
    '''


    #Retrieve CNN model names and number of training points 
    cnn_models = glob.glob('../models/SAR10_CNN_*.pt')
    cnn_num_train = [int(f[20:-3]) for f in cnn_models]

    #Sort models by number of training points
    I = np.argsort(cnn_num_train)
    cnn_num_train = [cnn_num_train[i] for i in I]
    cnn_models = [cnn_models[i] for i in I]
    cnn_train_idx = [cnn_models[i][:-3]+'_training_indices.npy' for i in range(len(I))]

    return cnn_models, cnn_train_idx, cnn_num_train

def NormalizeData(data):
    '''Normalizes data to range [0,1]

    Parameters
    ----------
    data : Numpy array

    Returns
    -------
    norm_data : Normalized array
    '''

    norm_data = (data - np.min(data))/(np.max(data) - np.min(data))
    return norm_data

def load_MSTAR(root_dir = '../Data'):
    """Loads MSTAR Data

    Parameters
    ----------
    root_dir : Root directory (default is ../Data)

    Returns
    -------
    hdr : header data
    fields : Names of fields in header data
    mag : Magnitude images
    phase : Phase images
    """

    M = np.load(os.path.join(root_dir,'SAR10a.npz'), allow_pickle=True)
    hdr_a,fields,mag_a,phase_a = M['hdr'],M['fields'],M['mag'],M['phase']

    M = np.load(os.path.join(root_dir,'SAR10b.npz'), allow_pickle=True)
    hdr_b,fields,mag_b,phase_b = M['hdr'],M['fields'],M['mag'],M['phase']

    M = np.load(os.path.join(root_dir,'SAR10c.npz'), allow_pickle=True)
    hdr_c,fields,mag_c,phase_c = M['hdr'],M['fields'],M['mag'],M['phase']

    hdr = np.concatenate((hdr_a,hdr_b,hdr_c))
    mag = np.concatenate((mag_a,mag_b,mag_c))
    phase = np.concatenate((phase_a,phase_b,phase_c))

    return hdr, fields, mag, phase

def train_test_split(hdr,train_fraction):
    '''Training and testing split (based on papers, angle=15 or 17)

    Parameters
    ----------
    hdr : Header info
    train_fraction : Fraction in [0,1] of full train data to use

    Returns
    -------
    full_train_mask : Boolean training mask for all angle==17 images
    test_mask : Boolean testing mask
    train_idx : Indices of training images selected
    '''

    angle = hdr[:,6].astype(int)
    full_train_mask = angle == 17
    test_mask = angle == 15
    num_train = int(np.sum(full_train_mask)*train_fraction)
    train_idx = np.random.choice(np.arange(hdr.shape[0]),size=num_train,replace=False,p=full_train_mask/np.sum(full_train_mask))

    return full_train_mask, test_mask, train_idx

def encodeMSTAR(model_path, batch_size = 1000, cuda = True, use_phase = False):
    '''Load a torch CNN model and encode MSTAR

    Parameters
    ----------
    model_path : Path to .pt file containing torch model for trained CNN
    batch_size : Size of minibatches to use in encoding. Reduce if you get out of memory errors (default = 1000)
    cuda : Whether to use GPU or not (default = True)
    use_phase : Whether the model uses phase information or not (default = False)

    Returns
    -------
    encoded_data : Returns a numpy array of MSTAR encoded by model.encode() (e.g., the CNN features)
    '''

    use_cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    #Load data and stack mag, real phase, and imaginary phase together
    hdr, fields, mag, phase = load_MSTAR()
    if use_phase:
        data = polar_transform(mag, phase)
    else:
        data = mag
    data = torch.from_numpy(data).float()

    #Load model
    model = torch.load(model_path)
    model.eval()
    encoded_data = None 
    with torch.no_grad():
        for idx in range(0,len(data),batch_size):
            data_batch = data[idx:idx+batch_size]
            if encoded_data is None:
                encoded_data = model.encode(data_batch.to(device)).cpu().numpy()
            else:
                encoded_data = np.vstack((encoded_data,model.encode(data_batch.to(device)).cpu().numpy()))

    return encoded_data


def targets_to_labels(hdr):
    '''Converts target names to numerical labels

    Parameters
    ----------
    hdr : Header data

    Returns
    -------
    labels : Integer labels from 0 to k-1 for k classes
    target_names : List of target names corresponding to each label integer
    '''

    targets = hdr[:,0].tolist()
    classes = set(targets)
    label_dict = dict(zip(classes, np.arange(len(classes))))
    labels = np.array([label_dict[t] for t in targets],dtype=int)
    target_names = list(label_dict.keys())

    return labels, target_names

def polar_transform(mag, phase):
    '''
    Peform polar transormation of data from Coman, Thomas.

    Parameters
    ----------
        mag : Magnitude images
        phase : Phase data

    Returns
    -------
        data : nx3 numpy array with (mag,real,imaginary)
    '''

    #obtain real and imaginary parts of phase data
    real_phase = NormalizeData(mag*np.cos(phase))
    imaginary_phase = NormalizeData(mag*np.sin(phase))
    data = np.stack((mag,real_phase,imaginary_phase),axis=1)

    return data

