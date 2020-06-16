# Import
import dicom2nifti
import dicom2nifti.settings as settings
import os
import numpy as np
import nibabel
import nibabel.orientations
import nilearn
from nilearn.image import smooth_img, math_img
import itertools
import fsl.wrappers.fslmaths as fslmaths
import math
import subprocess
import shutil

# Set directory of preprocessing folder
main_dirs = os.getcwd() + "/"

# Set directories of input and output folders
input_folder = main_dirs + "input/"
temp_folder = main_dirs + "temp/"
BET_folder = main_dirs
BET_input_folder = BET_folder + "image_data/"
BET_results_folder = BET_folder + "results_folder/"
output_folder = main_dirs + "output/"
input_folder_directory = os.listdir(input_folder)
temp_folder_directory = os.listdir(temp_folder)
BET_results_folder_directory = os.listdir(BET_results_folder)
weightFile = main_dirs + "mask_data/ct_20130524_1011_101232_j40s2.nii.gz"

# dcmniix settings
settings.enable_resampling()
settings.set_resample_spline_interpolation_order(1)
settings.set_resample_padding(-3024)
dicom2nifti.settings.disable_validate_orthogonal()

# Resampling parameters (Change this to desired dimenstions)
target_x_shape = 512
target_y_shape = 512
target_z_shape = 30

##### Functions #####

##### Functions #####

## (https://github.com/nipy/nibabel/issues/670) ##
def rescale_affine(input_affine, voxel_dims=[1, 1, 1], target_center_coords= None):
    """
    This function uses a generic approach to rescaling an affine to arbitrary
    voxel dimensions. It allows for affines with off-diagonal elements by
    decomposing the affine matrix into u,s,v (or rather the numpy equivalents)
    and applying the scaling to the scaling matrix (s).

    Parameters
    ----------
    input_affine : np.array of shape 4,4
        Result of nibabel.nifti1.Nifti1Image.affine
    voxel_dims : list
        Length in mm for x,y, and z dimensions of each voxel.
    target_center_coords: list of float
        3 numbers to specify the translation part of the affine if not using the same as the input_affine.

    Returns
    -------
    target_affine : 4x4matrix
        The resampled image.
    """
    # Initialize target_affine
    target_affine = input_affine.copy()
    # Decompose the image affine to allow scaling
    u,s,v = np.linalg.svd(target_affine[:3,:3],full_matrices=False)

    # Rescale the image to the appropriate voxel dimensions
    s = voxel_dims

    # Reconstruct the affine
    target_affine[:3,:3] = u @ np.diag(s) @ v

    # Set the translation component of the affine computed from the input
    # image affine if coordinates are specified by the user.
    if target_center_coords is not None:
        target_affine[:3,3] = target_center_coords
    return target_affine

def resample (file):
    for file in temp_folder_directory:
        print ("Loading " + file)
        img = nibabel.load(temp_folder + "/" + file)
        header = img.header

        # Get spatial_dimensions
        spatial_dimensions = (img.header['dim'] * img.header['pixdim'])[1:4]
        x_length = spatial_dimensions[0]
        y_length = spatial_dimensions[1]
        z_length = spatial_dimensions[2]

        vox_x_dim = x_length / target_x_shape
        vox_y_dim = y_length / target_y_shape
        vox_z_dim = z_length / target_z_shape

        # Getting affine
        target_affine = rescale_affine(img.affine, voxel_dims=[vox_z_dim, vox_y_dim, vox_x_dim])
        print ("Original affine")
        print (img.affine)
        print ("New affine")
        print (target_affine)

        # Resampling
        print ("Resampling")
        resampled_img = nilearn.image.resample_img(img, target_affine = target_affine, target_shape = (target_x_shape, target_y_shape, target_z_shape), interpolation='continuous', copy=True, order='F', clip=True, fill_value=0, force_resample=False)
        print ("New shape ")
        print (resampled_img.shape)
        nibabel.save(resampled_img, BET_input_folder + "resampled_img.nii.gz")


##### Running pipeline ######
for file in input_folder_directory:
# 1) Conversion of DICOM to NIFTI
    original_name = file
    print ("Converting ... " + (file))
    dicom_folder = input_folder + (file)
    dicom2nifti.convert_directory(dicom_folder, temp_folder, compression=True, reorient=True)
    
# 2) Resample image to shape (512, 512, 30)
    print ("Resampling ..." + (file))
    temp_folder_directory = os.listdir(temp_folder)
    resample(file)

# 3) Run BET
    print ("Running BET ...")
    subprocess.run(["python", main_dirs + "unet_CT_SS.py"])

# 4) Apply mask
    print ("Applying mask ...")
    BET_results_folder_directory = os.listdir(BET_results_folder)
    for file in BET_results_folder_directory:
        BET_prediction_folder = (BET_results_folder) + (file) + "/predictions/"
        BET_prediction_folder_directory = os.listdir(BET_prediction_folder)
        for file in BET_prediction_folder_directory:
            mask_file = nibabel.load(BET_prediction_folder + (file))
            original_file = nibabel.load(BET_input_folder + "resampled_img.nii.gz")
            BE_img = math_img('img1 * img2', img1=original_file, img2=mask_file)
            nibabel.save(BE_img, main_dirs + "resampled.nii.gz")

# 5) Cropping the image
    print ("Cropping image ...")
    img = smooth_img(main_dirs + "resampled.nii.gz",None)
    cropped_img = nilearn.image.crop_img(img, rtol=1e-08, copy=True, pad=True, return_offset=False)
    nibabel.save(cropped_img, main_dirs + "cropped_image.nii.gz")

# 6) Padding image to 512 x 512
    print ("Padding image ...")
    # Load image
    img = nibabel.load(main_dirs + "cropped_image.nii.gz")
    img_npy = img.get_data()
    
    # Get shape of original image
    (size_x, size_y, size_z) = img.shape
    
    # Calculate how many rows of zeros to be added
    pad_x = ((512-size_x)/2)
    pad_y = ((512-size_y)/2)
    pad_z = ((30-size_z)/2)
    
    # Use numpy.pad to add the rows of zeros to the file
    padded_img_npy = np.pad(img_npy, ((math.ceil(pad_x),math.floor(pad_x)), (math.ceil(pad_y),math.floor(pad_y)), (math.ceil(pad_z),math.floor(pad_z))), 'constant')
    
    # Save to .nii image
    img = nibabel.Nifti1Image(padded_img_npy, np.eye(4))
    nibabel.save(img, output_folder + (original_name) + ".nii.gz")

# 7) Clean up folders before next file
    BET_input_folder_directory = os.listdir(BET_input_folder)
    BET_results_folder_directory = os.listdir(BET_results_folder)
    os.unlink(main_dirs + "resampled.nii.gz")
    os.unlink(main_dirs + "cropped_image.nii.gz")
    for file in temp_folder_directory:
        os.unlink(temp_folder + file)
    for file in BET_input_folder_directory:
        os.unlink(BET_input_folder + file)
    for file in BET_results_folder_directory:
        shutil.rmtree(BET_results_folder + file)
