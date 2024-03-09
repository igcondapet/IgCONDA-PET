#%%
import pandas as pd 
import os 
from monai import transforms 
import sys 
from monai.utils import set_determinism
import random
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(WORKING_DIR)
from utils.utils import get_dict_datalist
from config import path_to_preprocessed_data_dir
#%%
set_determinism(42)

def get_transforms():
    mod_keys = ['PT', 'GT']
    data_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=mod_keys, image_only=False),
            transforms.EnsureChannelFirstd(keys=mod_keys),
            transforms.CopyItemsd(keys=['GT'], times=1, names=['Label']),
            transforms.Lambdad(keys=['Label'], func=lambda x: 2.0 if x.sum() > 0 else 1.0),
        ]
    )
    return data_transforms

def get_train_valid_datalist():
    metadata2D_fpath = os.path.join(WORKING_DIR, 'data_split', 'metadata2D.csv')
    metadata2D_df = pd.read_csv(metadata2D_fpath)
    train_df = metadata2D_df[metadata2D_df['TRAIN/VALID/TEST'] == 'TRAIN']
    train_df.reset_index(drop=True, inplace=True)
    valid_df = metadata2D_df[metadata2D_df['TRAIN/VALID/TEST'] == 'VALID']
    valid_df.reset_index(drop=True, inplace=True)

    # 30 patients will be removed from valid set. 
    # These 30 patients were used as a separate validation set for hyperparameter tuning
    # This random sampling has been seeded 
    # So during each run, the same 30 patients are excluded from the validation phase
    unique_valid_patientids = valid_df['PatientID'].unique().tolist()
    random.seed(0)
    sampled_patientids = random.sample(unique_valid_patientids, 30)
    valid_df = valid_df[~valid_df['PatientID'].isin(sampled_patientids)]
    valid_df.reset_index(drop=True, inplace=True)

    ptfnames_train, gtfnames_train = train_df['PTPATH'], train_df['GTPATH']
    ptfnames_valid, gtfnames_valid = valid_df['PTPATH'], valid_df['GTPATH']
    ptpaths_train = [os.path.join(path_to_preprocessed_data_dir, 'preprocessed2D', fname) for fname in ptfnames_train] 
    gtpaths_train = [os.path.join(path_to_preprocessed_data_dir, 'preprocessed2D', fname) for fname in gtfnames_train] 
    ptpaths_valid = [os.path.join(path_to_preprocessed_data_dir, 'preprocessed2D', fname) for fname in ptfnames_valid] 
    gtpaths_valid = [os.path.join(path_to_preprocessed_data_dir, 'preprocessed2D', fname) for fname in gtfnames_valid] 
    
    datalist_train = get_dict_datalist(ptpaths_train, gtpaths_train)
    datalist_valid = get_dict_datalist(ptpaths_valid, gtpaths_valid)
    
    return datalist_train, datalist_valid

#%%
def get_test_unhealthy_datalist():
    metadata2D_fpath = os.path.join(WORKING_DIR, 'data_split', 'metadata2D.csv')
    metadata2D_df = pd.read_csv(metadata2D_fpath)
    test_df = metadata2D_df[metadata2D_df['TRAIN/VALID/TEST'] == 'TEST']
    test_df.reset_index(drop=True, inplace=True)

    test_unhealthy_df = test_df[test_df['Label'] == 2]
    test_unhealthy_df.reset_index(drop=True, inplace=True)

    ptfnames, gtfnames = test_unhealthy_df['PTPATH'], test_unhealthy_df['GTPATH']
    ptpaths = [os.path.join(path_to_preprocessed_data_dir, 'preprocessed2D', fname) for fname in ptfnames] 
    gtpaths = [os.path.join(path_to_preprocessed_data_dir, 'preprocessed2D', fname) for fname in gtfnames] 

    datalist = get_dict_datalist(ptpaths, gtpaths)
    
    return datalist 

def get_train_valid_datalist_trial():
    from glob import glob 
    dir = '/home/jhubadmin/Projects/IgCONDA-PET/preprocessed_data/preprocessed2D'
    ptpaths = sorted(glob(os.path.join(dir, '*pt.nii.gz')))
    gtpaths = sorted(glob(os.path.join(dir, '*gt.nii.gz')))
    size = len(ptpaths)
    ptpaths_train, ptpaths_valid = ptpaths[:size//2], ptpaths[size//2:]
    gtpaths_train, gtpaths_valid = gtpaths[:size//2], gtpaths[size//2:]

    datalist_train = get_dict_datalist(ptpaths_train, gtpaths_train)
    datalist_valid = get_dict_datalist(ptpaths_valid, gtpaths_valid)
    return datalist_train, datalist_valid


def get_test_datalist_trial():
    from glob import glob 
    dir = '/home/jhubadmin/Projects/IgCONDA-PET/preprocessed_data/preprocessed2D'
    ptpaths = sorted(glob(os.path.join(dir, '*pt.nii.gz')))
    gtpaths = sorted(glob(os.path.join(dir, '*gt.nii.gz')))

    datalist = get_dict_datalist(ptpaths, gtpaths)
    return datalist