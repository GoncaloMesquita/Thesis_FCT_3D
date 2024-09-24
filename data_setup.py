
from tifffile import imread
import matplotlib.pyplot as plt
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
from scipy import stats
from utils import reconstruct_image_from_patches
from random import randint


def calculate_padding(image_shape, patch_size, overlap_size):

    # Calculate the total stride needed to cover the entire image
    stride_x = patch_size[0] - overlap_size[0]
    stride_y = patch_size[1] - overlap_size[1]
    stride_z = patch_size[2] - overlap_size[2]

    # Calculate the number of patches needed in each dimension
    num_patches_x = math.ceil(image_shape[0] / stride_x)
    num_patches_y = math.ceil(image_shape[1] / stride_y)
    num_patches_z = math.ceil(image_shape[2] / stride_z)

    # Calculate the total size covered by patches
    total_size_x = (num_patches_x - 1) * stride_x + patch_size[0]
    total_size_y = (num_patches_y - 1) * stride_y + patch_size[1]
    total_size_z = (num_patches_z - 1) * stride_z + patch_size[2]

    # Calculate padding needed to cover the remaining part of the image
    padding_x = max(0, total_size_x - image_shape[0])
    padding_y = max(0, total_size_y - image_shape[1])
    padding_z = max(0, total_size_z - image_shape[2])

    return padding_x, padding_y, padding_z


def create_patches(image, patch_size, overlap_percentage):

    overlap_size = [int(patch_dim * overlap_percentage / 100) for patch_dim in patch_size]
    padding_x, padding_y, padding_z = calculate_padding(image.shape[:-1], patch_size, overlap_size)

    padded_image = np.pad(image, (
        (padding_x ,0),
        (padding_y ,0),
        (padding_z ,0),
        (0, 0),
    ), mode='reflect')

    patches = []
    indices = []

    for i in range(0, padded_image.shape[0]  - (patch_size[0] ) +1, patch_size[0] - overlap_size[0]):
        for j in range(0, padded_image.shape[1] - (patch_size[1] ) +1 , patch_size[1] - overlap_size[1]):
            for k in range(0, padded_image.shape[2] - (patch_size[2] ) +1 , patch_size[2] - overlap_size[2]):
                
                patch = padded_image[i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]]
                patches.append(patch)
                indices.append((i, j, k))

    return patches, indices, padded_image, (padding_x, padding_y, padding_z)


def get_number_from_filename(filename):

    numeric_part = ''.join(filter(str.isdigit, filename))
    return int(numeric_part) if numeric_part else 0


def create_dataset(dir1, dir2, patch_size, in_channels, overlap):

    image_files = [f for f in os.listdir(dir1) if f.startswith("Crop") ]
    mask_files = [f for f in os.listdir(dir2) if f.startswith("Crop") ]

    image_files = sorted(image_files, key=get_number_from_filename)
    mask_files = sorted(mask_files, key=get_number_from_filename)

    n_images = len(image_files)

    images = []
    masks = []
    original_image = []
    original_mask = []  
    paddings =[]
    overlap_percentage = overlap
    n_patches = np.zeros(n_images)

    print("Downloading Dataset ...")
    # fig , axis = plt.subplots(1, 2, figsize=(7, 10))
    # fig.suptitle('Class Weight Histogram', fontsize=16)
    # m = 0
    # j = 0 
    
    for idx in range(0, n_images):
        
        image_path = os.path.join(dir1, image_files[idx])
        mask_path = os.path.join(dir2, mask_files[idx])

        image = imread(image_path)
        mask = imread(mask_path)
        
        patches, indices, image_padded, padding  = create_patches(image[:, :, :, 0:in_channels], patch_size, overlap_percentage)
        paddings.append(padding)
        original_image.append(image_padded)
        n_patches[idx] = len(patches)
        images.append((patches, indices, image_padded.shape, n_patches[idx]))
        
        patches, indices, mask_padded, _ = create_patches(mask[:, :, :, 0:in_channels], patch_size, overlap_percentage)
        original_mask.append(mask_padded)
        masks.append((patches, indices, mask_padded.shape, n_patches[idx]))
        
    # plt.tight_layout(rect=[0, 0.05, 1, 0.95]) 
    # plt.show()
    # Flatten lists of images and masks
    flat_images = [item for sublist in images for item in sublist]
    flat_masks = [item for sublist in masks for item in sublist]

    images = np.vstack(flat_images[::4])
    masks = np.vstack(flat_masks[::4])

    split = [ (images[:int(n_patches[0:6].sum())], masks[:int(n_patches[0:6].sum())], images[int(n_patches[0:6].sum()):], masks[int(n_patches[0:6].sum()):]),  
    (np.concatenate((images[:int(n_patches[0:4].sum())], images[int(n_patches[0:6].sum()):])), np.concatenate((masks[:int(n_patches[0:4].sum())], masks[int(n_patches[0:6].sum()):])),images[int(n_patches[0:4].sum()):int(n_patches[0:6].sum())], masks[int(n_patches[0:4].sum()):int(n_patches[0:6].sum())]),  
    (images[int(n_patches[0:5].sum()):], masks[int(n_patches[0:5].sum()):], images[:int(n_patches[0:4].sum())], masks[:int(n_patches[0:4].sum())])]
    
    return split, original_image, original_mask, flat_images, paddings


def dataset_generalized(dir1, dir2, patch_size, in_channels, overlap):
    
    dir3 = 'Dataset/Images_Generalize/'
    
    image_files = [f for f in os.listdir(dir1) if f.startswith("Crop") ]
    mask_files = [f for f in os.listdir(dir2) if f.startswith("Crop") ]
    test_files = [f for f in os.listdir(dir3) if f.endswith("tif") ]

    image_files = sorted(image_files, key=get_number_from_filename)
    mask_files = sorted(mask_files, key=get_number_from_filename)

    n_images = len(image_files)

    images = []
    masks = []
    test_images = []

    test_original_image = []
    paddings =[]
    overlap_percentage = overlap
    n_patches = np.zeros(n_images)

    print("Downloading Dataset ...")
    
    for idx in range(0, n_images):
        
        image_path = os.path.join(dir1, image_files[idx])
        mask_path = os.path.join(dir2, mask_files[idx])

        image = imread(image_path)
        mask = imread(mask_path)
        
        patches, indices, image_padded, padding  = create_patches(image[:, :, :, 0:in_channels], patch_size, overlap_percentage)
        images.append(patches)
        
        patches, indices, mask_padded, _ = create_patches(mask[:, :, :, 0:in_channels], patch_size, overlap_percentage)
        masks.append(patches)
        
    for idx in range(0,1):

        test_path = os.path.join(dir3, test_files[idx])
        test = imread(test_path)
        
        patches, indices, test_padded, padding  = create_patches(test[:, :, :, 0:in_channels], patch_size, overlap_percentage)
        
        paddings.append(padding)
        test_original_image.append(test_padded)
        n_patches[idx] = len(patches)
        test_images.append((patches, indices, test_padded.shape, n_patches[idx]))
        
        
    flat_images_test = [item for sublist in test_images for item in sublist]
    images = np.vstack(images)
    masks = np.vstack(masks)
    test_images = np.vstack(flat_images_test[::4])
    

    test_original_mask = [np.zeros_like(array) for array in test_original_image]
    test_mask = np.zeros(test_images.shape)
    
    split = [(images, masks, test_images[50:51], test_mask[50:51])]
    
    
    
    # return split, test_original_image, test_original_mask, flat_images_test, paddings
    return split, test_original_image, test_original_mask, flat_images_test, paddings
    


    # print(images.shape)
    # print(masks.shape)

    # stats = {
    #     'mean': np.mean(image[:,:,1]),
    #     'std': np.std(image[:,:,1]),
    #     'min': np.min(image[:,:,1]),
    #     '25%': np.percentile(image[:,:,1], 25),
    #     '50%': np.median(image[:,:,1]),
    #     '75%': np.percentile(image[:,:,1], 75),
    #     'max': np.max(image[:,:,1]),
    #     'value_counts': dict(zip(*np.unique(image[:,:,1], return_counts=True)))
    # }
    
    # for key, value in stats.items():
    #     print(f"{key}: {value}")
    # plt.imshow(masks[245,30, :,:,1])
    # plt.show
       #     axis[m,j].hist(image[:,:,:,0].flatten(), bins=25, alpha=0.5,  color='red')
    #     # axis[m, j].set_yscale('log')
    #     # ax.axis('off')
    #     stats = {
    #     'mean': np.mean(image[:,:,:,0]),
    #     'std': np.std(image[:,:,:,0]),
    #     'min': np.min(image[:,:,:,0]),
    #     '25%': np.percentile(image[:,:,:,0], 25),
    #     '50%': np.median(image[:,:,:,0]),
    #     '75%': np.percentile(image[:,:,:,0], 75),
    #     'max': np.max(image[:,:,:,0]),
    #     # Optional: 'value_counts': dict(zip(*np.unique(image[:,:,1], return_counts=True)))
    #     }
    #     # legend_text = '\n'.join([f'{key}: {value:.2f}' for key, value in stats.items()])
    #     # axis[m,j].legend(title='Statistics', title_fontsize='small', labels=[legend_text], handlelength=0, handletextpad=0, fancybox=True)
        
    #     # for stat_name, stat_value in stats.items():
    #     #     if stat_name == 'mean':
    #     #         axis[m,j].axvline(stat_value, color='blue', linestyle='-', label='Mean')
    #     #     elif stat_name == 'std':
    #     #         axis[m,j].axvline(stat_value, color='green', linestyle='--', label='standard deviation')
    #     #     elif stat_name == 'min':
    #     #         axis[m,j].axvline(stat_value, color='cyan', linestyle=':', label='Minimum')
    #     #     elif stat_name == '25%':
    #     #         axis[m,j].axvline(stat_value, color='magenta', linestyle='-.', label='25% Percentile')
    #     #     elif stat_name == '50%':  # Median
    #     #         axis[m,j].axvline(stat_value, color='yellow', linestyle='-', label='Median')
    #     #     elif stat_name == '75%':
    #     #         axis[m,j].axvline(stat_value, color='black', linestyle='--', label='75% Percentile')
    #     #     elif stat_name == 'max':
    #     #         axis[m,j].axvline(stat_value, color='orange', linestyle=':', label='Maximum')
        
    # # Add a legend
    #     # axis[m,j].legend()
    #     axis[m,j].set_title(f'Image {idx+1}')
    #     axis[m,j].set_xlabel('Intensity Value')
    #     axis[m,j].set_ylabel('Pixel Count')
    #     axis[m,j].set_xlim([0, stats['max']+4])
    #     j += 1
    #     if j == 4:
    #         j = 0
    #         m = 1
