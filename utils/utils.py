#%%
import argparse
import SimpleITK as sitk 

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def convert_to_N_digits(num, N):
    return str(num).zfill(N)

def get_dict_datalist(ptpaths, gtpaths):
    datalist = []
    for i in range(len(gtpaths)):
        datalist.append({'PT':ptpaths[i], 'GT': gtpaths[i]})
    return datalist  

def read_image_array(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path)).T

def save_image(array, savepath):
    image = sitk.GetImageFromArray(array.T)
    sitk.WriteImage(image, savepath)
