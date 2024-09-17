import os
from glob import glob
import shutil
from tqdm import tqdm
# import dicom2nifti
import numpy as np
# import nibabel as nib
from monai.transforms import(
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
)
from monai.data import DataLoader, Dataset, CacheDataset
from monai.utils import set_determinism

"""
PREPROCESS
"""

def create_groups(in_dir, out_dir, Number_slices):


    for patient in glob(in_dir + '/*'):
        patient_name = os.path.basename(os.path.normpath(patient))

        # Here we need to calculate the number of folders which mean into how many groups we will divide the number of slices
        number_folders = int(len(glob(patient + '/*')) / Number_slices)

        for i in range(number_folders):
            output_path = os.path.join(out_dir, patient_name + '_' + str(i))
            os.mkdir(output_path)

            # Move the slices into a specific folder so that you will save memory in your desk
            for i, file in enumerate(glob(patient + '/*')):
                if i == Number_slices + 1:
                    break
                
                shutil.move(file, output_path)




# def dcm2nifti(in_dir, out_dir):
#     for folder in tqdm(glob(in_dir + '/*')):
#         patient_name = os.path.basename(os.path.normpath(folder))
#         dicom2nifti.dicom_series_to_nifti(folder, os.path.join(out_dir, patient_name + '.nii.gz'))


# def find_empy(in_dir):
#     list_patients = []
#     for patient in glob(os.path.join(in_dir, '*')):
#         img = nib.load(patient)

#         if len(np.unique(img.get_fdata())) > 2:
#             print(os.path.basename(os.path.normpath(patient)))
#             list_patients.append(os.path.basename(os.path.normpath(patient)))
    
#     return list_patients


def prepare(in_dir, pixdim=(1.5, 1.5, 1.0), a_min=-200, a_max=200, spatial_size=[128,128,64], cache=False):

    set_determinism(seed=0)

    def sort_by_liver_number(file_path):
        # Extract the number from the file name
        file_name = os.path.basename(file_path)
        liver_number = int(file_name.split('_')[1].split('.')[0])
        return liver_number

    folders = glob(in_dir +'/*')

    training_images = sorted(glob(in_dir + '/ImagesTr' + "/*" ))
    training_labels = sorted(glob(in_dir + '/LabelsTr' + "/*" ))


    training_images = sorted(training_images, key=sort_by_liver_number)[:99]
    training_labels = sorted(training_labels, key=sort_by_liver_number)[:99]

    testing_images = sorted(training_images, key=sort_by_liver_number)[100:]
    testing_labels = sorted(training_labels, key=sort_by_liver_number)[100:]

    train_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in zip(training_images, training_labels)]
    test_files = [{'vol': image_name, "seg": label_name} for image_name, label_name in zip(testing_images, testing_labels)]
    


    train_transforms = Compose(
        [
            LoadImaged(keys=["vol", "seg"]),
            EnsureChannelFirstd(keys=["vol", "seg"]),
            Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
            Orientationd(keys=["vol", "seg"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True), 
            CropForegroundd(keys=["vol", "seg"], source_key="vol"),
            Resized(keys=["vol", "seg"], spatial_size=spatial_size),   
            ToTensord(keys=["vol", "seg"]),

        ]
    )


    test_transforms = Compose(
        [
            LoadImaged(keys=["vol", "seg"]),
            EnsureChannelFirstd(keys=["vol", "seg"]),
            Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
            Orientationd(keys=["vol", "seg"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max,b_min=0.0, b_max=1.0, clip=True), 
            CropForegroundd(keys=['vol', 'seg'], source_key='vol', allow_smaller=False),
            Resized(keys=["vol", "seg"], spatial_size=spatial_size),   
            ToTensord(keys=["vol", "seg"]),

            
        ]
    )
    

    if cache:
        train_ds = CacheDataset(data=train_files, transform=train_transforms,cache_rate=1.0)
        train_loader = DataLoader(train_ds, batch_size=1)

        test_ds = CacheDataset(data=test_files, transform=test_transforms, cache_rate=1.0)
        test_loader = DataLoader(test_ds, batch_size=1)

        return train_loader, test_loader

    else:
        train_ds = Dataset(data=train_files, transform=train_transforms)
        train_loader = DataLoader(train_ds, batch_size=1)

        test_ds = Dataset(data=test_files, transform=test_transforms)
        test_loader = DataLoader(test_ds, batch_size=1)

        return train_loader, test_loader


train_loader, test_loader = prepare(in_dir='/Users/tylerklimas/Desktop/LiverSegmentation/Task03_liver')


