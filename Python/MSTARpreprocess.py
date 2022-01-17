'''
This script is for preprocessing the MSTAR data from the raw format 
that you can download from the MSTAR website 

https://www.sdms.afrl.af.mil/index.php?collection=mstar

You must register a free account in order to download the data.
There are several different zip files available. This script
needs the following 3 zip files, and they should be placed
at the locations below:

../Data/MSTAR-PublicMixedTargets-CD1.zip
../Data/MSTAR-PublicMixedTargets-CD2.zip
../Data/MSTAR-PublicTargetChips-T72-BMP2-BTR70-SLICY.zip

The total size of the 3 zip files is around 900MB.

The script crops the magnitude and phase images to 88x88
pixel images and saves the images and header information 
to the files

../Data/SAR10a.npz
../Data/SAR10b.npz
../Data/SAR10c.npz

The reason for 3 files instead of 1 is to ensure the file
sizes are below the GitHub maximum of 100MB. These files
are provided in GitHub, so this script does not need to be run,
and is solely provided for reproducibility of the code.

NOTE: This script requires a compiled binary for the c program
mstar2raw. The c code is provided in the GitHub repository, in 
the directory mstar2raw. Make sure to compile this on your machine.
The compiled binary should be located at

mstar2raw/mstar2raw

The GitHub package includes a binary compiled on a machine running
Ubuntu Linux.
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import zipfile

def crop(img,numrows,numcols,size):
    '''Center crop an image from a container image
    
    Parameters
    ----------
    img : Numpy array for grayscale container image
    numrows : Number of rows in subimage
    numcols : Number of columns in subimage
    size : Crop size (square crop)

    Returns
    -------
    cropped_img : Cropped image
    '''

    s0 = (numrows - size)//2
    s1 = (numcols - size)//2
    cropped_img = img[s0:s0+size,s1:s1+size]
    return cropped_img


def read_hdr(fname):
    '''Read header file (.hdr) for MSTAR image

    Parameters
    ----------
    fname : Base filename (without .hdr)

    Returns
    -------
    hdr_fields : Tuple of field names and data values for the header information ('Filename','NumberOfColumns','NumberOfRows','TargetType','TargetSerNum','TargetAz','TargetRoll','TargetPitch','TargetYaw','DesiredDepression')
    data : Tuple of data corresponding to the hdr_fields
    '''

    #Read header into python to get image dimensions
    df = pd.read_csv(fname+'.hdr',delimiter='=')
    df = df.transpose()
    NumberOfColumns = int(df['NumberOfColumns'][0])
    NumberOfRows = int(df['NumberOfRows'][0])
    TargetType = df['TargetType'][0].strip()
    TargetSerNum = df['TargetSerNum'][0].strip()
    TargetAz = float(df['TargetAz'][0])
    TargetRoll = float(df['TargetRoll'][0])
    TargetPitch = float(df['TargetPitch'][0])
    TargetYaw = float(df['TargetYaw'][0])
    DesiredDepression = int(df['DesiredDepression'][0])

    hdr_fields = ('Filename','NumberOfColumns','NumberOfRows','TargetType','TargetSerNum','TargetAz','TargetRoll','TargetPitch','TargetYaw','DesiredDepression') 
    data = (fname,NumberOfColumns,NumberOfRows,TargetType,TargetSerNum,TargetAz,TargetRoll,TargetPitch,TargetYaw,DesiredDepression)

    return hdr_fields,data

def read_images(fname,numrows,numcols):
    '''Read magnitude and phase images from .all raw file
    
    Parameters
    ----------
    fname : File name
    numrows : Number of pixel rows
    numcols : Number of pixel columns

    Returns
    -------
    mag : Magnitude image
    phase : Phase image
    '''

    #Read mag and phase images
    img = np.fromfile(fname+'.all',dtype=np.float32)
    m = numrows*numcols
    mag = np.reshape(img[:m],(numrows,numcols))
    phase = np.reshape(img[m:],(numrows,numcols))

    return mag,phase


#Read all images and header files in a given directory
def read_dir(root,count):
    '''Read all images and header files in a given directory
       This function writes the data into the global variables
       hdr_data, mag_data, and phase_data.

    Parameters
    ----------
    root : Root directory to recursively read all images from. 
    count : Current count of how many have images have been read

    Returns
    -------
    num : Number of images read from this directory
    fields : Tuple of header fields
    '''
    num = 0
    for subdir, directories, files in os.walk(root):
        for file in files:
            if file.endswith('.all'):

                full_name = os.path.join(subdir,file[:-4])
                if not '/COL2/SCENE1/SLICY/' in full_name:
                    print('Reading ' + full_name + '...')

                    #Read header
                    fields,data=read_hdr(full_name)
                    numcols = data[1]
                    numrows = data[2]

                    #Read images
                    mag,phase = read_images(full_name,numrows,numcols)

                    #Store header and images into numpy arrays
                    hdr_data[count,:] = np.array(data)
                    mag_data[count,:numrows,:numcols] = mag
                    phase_data[count,:numrows,:numcols] = phase

                    count += 1
                    num += 1
    return num,fields


MSTAR_dir = '../Data'
zipfiles = ['MSTAR-PublicMixedTargets-CD1.zip',
            'MSTAR-PublicMixedTargets-CD2.zip',
            'MSTAR-PublicTargetChips-T72-BMP2-BTR70-SLICY.zip']

#Unzip MSTAR Files
print('Unzipping MSTAR data...')
for file in zipfiles:
    with zipfile.ZipFile(os.path.join(MSTAR_dir,file), 'r') as zip_ref:
        zip_ref.extractall(MSTAR_dir)

#Convert all MSTAR files to raw format
cwd = os.getcwd()
mstar2raw = os.path.join(cwd,'../mstar2raw','mstar2raw')
for subdir, dirs, files in os.walk(MSTAR_dir):
    for file in files:
        if file.startswith('H') and (not file.endswith(('JPG','all','hdr','mag'))):
            os.chdir(subdir)
            os.system(mstar2raw + " " + file + " 0")
            os.chdir(cwd)

#Directories after unzipping
dirs = ['MSTAR_PUBLIC_TARGETS_CHIPS_T72_BMP2_BTR70_SLICY/TARGETS/',
        'MSTAR_PUBLIC_MIXED_TARGETS_CD1/15_DEG',
        'MSTAR_PUBLIC_MIXED_TARGETS_CD2/17_DEG']

#Pre-allocate memory for faster reading
hdr_len = 10
max_rows = 193
max_cols = 192
max_len = 10000
mag_data = np.zeros((max_len,max_rows,max_cols))
phase_data = np.zeros((max_len,max_rows,max_cols))
hdr_data = np.empty((max_len,hdr_len),dtype=object)

count = 0
#Read all data
for d in dirs:
    num,fields = read_dir(os.path.join(MSTAR_dir,d),count)
    count += num

#Restrict to data that was read
hdr_data = hdr_data[:count]
mag_data = mag_data[:count]
phase_data = phase_data[:count]

#Now we crop to 88x88 pixels
print('Cropping...')
size = 88
mag_cropped = np.zeros((mag_data.shape[0],size,size))
phase_cropped = np.zeros((phase_data.shape[0],size,size))
for i in range(mag_data.shape[0]):
    numrows = int(hdr_data[i,1])
    numcols = int(hdr_data[i,2])

    mag_cropped[i,:,:] = crop(mag_data[i,:,:],numrows,numcols,size)
    phase_cropped[i,:,:] = crop(phase_data[i,:,:],numrows,numcols,size)

#Remove filename and image shape from header
hdr_data = hdr_data[:,3:]
fields = fields[3:]

#Save Cropped Data, split into 3 separate files
print('Saving to npz files...')
n = count//3
np.savez_compressed(os.path.join(MSTAR_dir,'SAR10a.npz'),hdr=hdr_data[:n,:],mag=mag_cropped[:n,:,:],phase=phase_cropped[:n,:,:],fields=fields)
np.savez_compressed(os.path.join(MSTAR_dir,'SAR10b.npz'),hdr=hdr_data[n:2*n,:],mag=mag_cropped[n:2*n,:,:],phase=phase_cropped[n:2*n,:,:],fields=fields)
np.savez_compressed(os.path.join(MSTAR_dir,'SAR10c.npz'),hdr=hdr_data[2*n:,:],mag=mag_cropped[2*n:,:,:],phase=phase_cropped[2*n:,:,:],fields=fields)


