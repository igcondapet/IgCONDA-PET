#%%
import pandas as pd 
import os 
import sys 
from monai.data import CacheDataset
from monai import transforms
import numpy as np 
from glob import glob 
from monai import transforms
import SimpleITK as sitk 
from joblib import Parallel, delayed
import shutil
import time 
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(WORKING_DIR)
from config import path_to_hecktor_data_dir, path_to_autopet_data_dir, path_to_preprocessed_data_dir
from utils.utils import convert_to_N_digits, get_dict_datalist, read_image_array
#%%
# %%
def save_2Dimages_from_indices(patientid, list_of_indices, ptarray, gtarray, savedir):
    for i in list_of_indices:
        pt_2d = ptarray[:, :, i].T
        gt_2d = gtarray[:, :, i].T
        pt2dimage = sitk.GetImageFromArray(pt_2d)
        gt2dimage = sitk.GetImageFromArray(gt_2d)
        sliceid = convert_to_N_digits(i, 3)
        pt2dsavepath = os.path.join(savedir, f'{patientid}_{sliceid}_pt.nii.gz')
        gt2dsavepath = os.path.join(savedir, f'{patientid}_{sliceid}_gt.nii.gz')
        sitk.WriteImage(pt2dimage, pt2dsavepath)
        sitk.WriteImage(gt2dimage, gt2dsavepath)

# %%
preprocessed3D_data_dir = os.path.join(path_to_preprocessed_data_dir, 'preprocessed3D')
os.makedirs(preprocessed3D_data_dir, exist_ok=True)
preprocessed3D_images_dir = os.path.join(preprocessed3D_data_dir, 'images')
preprocessed3D_labels_dir = os.path.join(preprocessed3D_data_dir, 'labels')
os.makedirs(preprocessed3D_images_dir, exist_ok=True)
os.makedirs(preprocessed3D_labels_dir, exist_ok=True)
# %%
metadata3D_fpath = os.path.join(WORKING_DIR, 'data_split', 'metadata3D.csv')
df3D = pd.read_csv(metadata3D_fpath)

ptpaths3D, gtpaths3D = [], []
for index, row in df3D.iterrows():
    dataset = row['Dataset']
    if dataset == 'autopet':
        dataset_dir = path_to_autopet_data_dir
    elif dataset == 'hecktor':
        dataset_dir = path_to_hecktor_data_dir
    else:
        print('Incorrect dataset!')

    ptpath = os.path.join(dataset_dir, 'images', row['PTPATH'])
    gtpath = os.path.join(dataset_dir, 'labels', row['GTPATH'])
    ptpaths3D.append(ptpath)
    gtpaths3D.append(gtpath)

datalist3D = get_dict_datalist(ptpaths3D[0:20], gtpaths3D[0:20])
#%%
mod_keys = ['PT', 'GT'] 
autopet_spacing = (2.0364201068878174, 2.0364201068878174, 3.0)
resampling_mode =  ['bilinear', 'nearest']
crop_dim = [192, 192, 288]
resampled_dim = [64, 64, 96]
data_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=mod_keys, image_only=False),
            transforms.EnsureChannelFirstd(keys=mod_keys),
            transforms.Spacingd(keys=mod_keys, pixdim=autopet_spacing, mode=resampling_mode),
            transforms.CenterSpatialCropd(keys=mod_keys, roi_size=crop_dim),
            transforms.ScaleIntensityd(keys=['PT']),
            transforms.Resized(keys=mod_keys, spatial_size=resampled_dim, mode=resampling_mode),
            transforms.SqueezeDimd(keys=mod_keys, dim=0),
            transforms.SaveImaged(keys=['PT'], output_dir = preprocessed3D_images_dir, output_postfix='', separate_folder=False),
            transforms.SaveImaged(keys=['GT'], output_dir = preprocessed3D_labels_dir, output_postfix='', separate_folder=False)
        ]
    )

# %%
print(f'Downsampling 3D images to dimensions: {resampled_dim}. This may take a while!')
print(f'The downsampled 3D images will be saved in: {preprocessed3D_data_dir}\n')
start = time.time()
dataset = CacheDataset(datalist3D, transform=data_transforms, cache_rate=1.0)
elapsed_time = time.time() - start 
print(f'\nDownsampling process took {elapsed_time/60:.2f} mins\n')


# %%
preprocessed2D_data_dir = os.path.join(path_to_preprocessed_data_dir, 'preprocessed2D')
os.makedirs(preprocessed2D_data_dir, exist_ok=True)

ptpaths3D_downsampled = sorted(glob(os.path.join(preprocessed3D_images_dir, '*.nii.gz')))
gtpaths3D_downsampled = sorted(glob(os.path.join(preprocessed3D_labels_dir, '*.nii.gz')))

# %%
def process_patient(ptpath, gtpath, preprocessed2D_data_dir):
    patientid = os.path.basename(gtpath)[:-7]
    ptarray = read_image_array(ptpath)
    gtarray = read_image_array(gtpath)
    indices = np.arange(0,96)
    save_2Dimages_from_indices(patientid, indices, ptarray, gtarray, preprocessed2D_data_dir)
    print(f'Done saving all 2D slices for patientID: {patientid}')

#%%
start = time.time()
print('\nSaving 2D slices (64x64) for all downsampled images. This may take a little while!')
print(f'The 2D slices will be saved in: {preprocessed2D_data_dir}\n')
_ = Parallel(n_jobs=5)(delayed(process_patient)(p, g, preprocessed2D_data_dir) for i, (p, g) in enumerate(zip(ptpaths3D_downsampled, gtpaths3D_downsampled)))
elapsed_time = time.time() - start 
print(f'\nSaving 2D slices for all images took {elapsed_time/60:.2f} mins')
# %%
