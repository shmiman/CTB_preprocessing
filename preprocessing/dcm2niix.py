# Import necessary dcm2niix python wrapper
import dicom2nifti
import dicom2nifti.settings as settings

# Set settings for conversion
settings.enable_resampling()
settings.set_resample_spline_interpolation_order(1)
settings.set_resample_padding(-3024)
dicom2nifti.settings.disable_validate_orthogonal()

# Get directory of .dcm folder
import os
import shutil

dirs = os.listdir("D:\MDRP/train/")

# I'm going to output the file to a temporary folder because I can't rename the file
# The script names the file as the modality instead of original folder name and I can't find how to change it in the python wrapper
temp_folder = 'D:\MDRP/preprocessing/temp/'

# This is the main output folder where we want our correctly named files 
output_folder = 'D:\MDRP/preprocessing/nifti_output/'

for file in dirs:
    try:
        original_file = file

        # First we list the directories
        dicom_directory = 'D:\MDRP/train/' + (original_file)

        # Converting a file in the dcm directory to the temp folder
        dicom2nifti.convert_directory(dicom_directory, temp_folder, compression=True, reorient=True)

        temp_dirs = os.listdir("/D:\MDRP/preprocessing/temp/")
        for file in temp_dirs:
            temp_file = file

        # Renaming the file in the temp dirs to the original file name and transferring the directory
        os.rename(temp_folder + "/" + (temp_file), output_folder + "/" + (original_file) + ".nii.gz")

        # Deleting contents of temp folder
        shutil.rmtree(temp_folder)
        os.mkdir(temp_folder)
    except:
        print (file)
        pass
