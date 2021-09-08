import numpy as np
import torch
import os


def load_MSTAR(root_dir = 'data'):
    """
    Loads MSTAR Data

    Returns:
    header, fields, magnitude images, phase images
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


def polar_transform(mag, phase):
    """
    Peform polar transormation of data from Coman, Thomas.
    Inputs:
        mag - magnitude data
        phase - phase data
    Returns:
        scale_real_phase - scaled real phase data
        scale_imaginary_phase - scaled imaginary phase data
    """
    mag_data = torch.from_numpy(np.reshape(mag,(mag.shape[0],1,mag.shape[1],mag.shape[2]))).float()
    phase_data = torch.from_numpy(np.reshape(phase,(phase.shape[0],1,phase.shape[1],phase.shape[2]))).float()

    # Calculate max and min of magnitude data
    mag_min = torch.min(mag_data.flatten()).item()
    mag_max = torch.max(mag_data.flatten()).item()

    #scale magnitude data to be between 0 and 1
    scale_mag_data = (mag_data.flatten() - mag_min)/ (mag_max - mag_min)

    #take cosine and sine functions of phase data
    cosine_phase_data = torch.cos(phase_data.flatten())
    sine_phase_data = torch.sin(phase_data.flatten())

    #obtain real and imaginary parts of phase data
    real_phase = scale_mag_data*cosine_phase_data
    imaginary_phase = scale_mag_data*sine_phase_data

    #calculate minimum value of real and imaginary phase data
    real_phase_min = torch.min(real_phase).item()
    imaginary_phase_min = torch.min(imaginary_phase).item()

    #translate into positive domain values if necessary
    if real_phase_min < 0:
        real_phase -= real_phase_min

    if imaginary_phase_min < 0:
        imaginary_phase -= imaginary_phase_min

        #calculate min and max of real and imaginary phase data
    real_phase_min = torch.min(real_phase).item()
    real_phase_max = torch.max(real_phase).item()
    imaginary_phase_min = torch.min(imaginary_phase).item()
    imaginary_phase_max = torch.max(imaginary_phase).item()

    #scale phase data to be between 0 and 1
    scale_real_phase = (real_phase - real_phase_min) / (real_phase_max - real_phase_min)
    scale_imaginary_phase = (imaginary_phase - imaginary_phase_min) / (imaginary_phase_max -imaginary_phase_min)

    #reshape back into original shape
    scale_real_phase = scale_real_phase.reshape((6874, 1, 88, 88))
    scale_imaginary_phase = scale_imaginary_phase.reshape((6874, 1, 88, 88))

    return scale_real_phase,scale_imaginary_phase
