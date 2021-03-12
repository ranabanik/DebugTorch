import os
from glob import glob
import nibabel as nib
import numpy as np

data_dir = r'C:\Data'

filepath = glob(os.path.join(data_dir, '*.gz'))
img = nib.load(filepath[0])
data = img.get_fdata()
print(data.max(), data.min())
top_thresh = np.percentile(data, 97.5)
bot_thresh = np.percentile(data, 2.5)
# print(top_97_thresh)
new_data = data.copy()
new_data[new_data > top_thresh] = top_thresh
new_data[new_data < bot_thresh] = bot_thresh
print(new_data.max(), new_data.min())  # doctest: +SKIP
clipped_img = nib.Nifti1Image(new_data, img.affine, img.header)
# print(type(clipped_img))

# empty_header = nib.Nifti1Header()
# another_img = nib.Nifti1Image(new_data, img.affine, empty_header)
new_filepath = os.path.join(data_dir, 'clipped_image.nii')
nib.save(clipped_img, new_filepath)