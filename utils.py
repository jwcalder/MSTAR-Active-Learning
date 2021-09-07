import numpy as np
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
