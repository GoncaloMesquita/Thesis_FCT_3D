import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
#  Full_images_96 , (Image, Mask, 3D U-Net, UNETR, FCT) , 1 6 8 , 'a)', 'b)', 'c)', 
#  Full_images_256 , (Image, Mask, 3D U-Net, UNETR, FCT) , 1 6 8 , 'a)', 'b)', 'c)',
#  Full_images_256/low_contrast , (Image, Mask, 3D U-Net, UNETR, FCT) , 8 ,"d)"
#  Full_images_256/noise , (Image, Mask, 3D U-Net, UNETR, FCT) ,  8 , "e)"
# Generalization/Image_1 , (Image, 3D U-Net, UNETR, FCT) , 10 Low_p6  , 'a)', 'b)'
# Generalization/Image_1/low_contrast , (Image, 3D U-Net, UNETR, FCT) , 10 Low_p6 , a-1) b-1)
# Generalization/Image_1/pair_nuclei-golgi , (Image, 3D U-Net, UNETR, FCT) , icam icam2, a-2) b-2)
# Generalization/Image_2 , (Image, 3D U-Net, UNETR, FCT) , icam icam2  , 'c)', 'd)'
# Generalization/Image_2/nuclei , (Image, 3D U-Net, UNETR, FCT) , icam icam2 , c-1) d-1)
# Generalization/Image_2/Golgi , (Image, 3D U-Net, UNETR, FCT) , icam icam2 , c-2) d-2)
# Generalization/p6_avm , (Image, 3D U-Net, UNETR, FCT) , p6_avm  , 'e)'
# Generalization/p6_avm/irregular_structures_1 , (Image, 3D U-Net, UNETR, FCT) , p6_avm p6_avm2, e-1)
# Generalization/p6_avm/irregular_structures_2 , (Image, 3D U-Net, UNETR, FCT) , p6_avm p6_avm2, e-2)

# # strings_y = []
# ############################# IMAGES FROM FOLDER ##############################
# # Define the folder where images are stored
# folder_path = 'Generalization/p6_avm/irregular_structures_1'
# # Define the image and mask types you need to plot
# image_ids = ["p6_avm", "p6_avm_2"]
# methods = ['Image', '3D U-Net', 'UNETR', 'FCT']
# strings_y = []
# # Set up the plot
# fig, axes = plt.subplots(nrows=len(image_ids), ncols=len(methods), figsize=(12,6))

# axes = np.array(axes)  # Ensure axes is always 2-dimensional
# axes = axes.reshape(len(image_ids), -1)  # Reshape to 2D if necessary

# # Style adjustments for a more professional look
# plt.subplots_adjust(hspace=0.05, wspace=0.05)
# for ax in axes.flatten():
#     ax.axis('off')

# # Load and display each image according to the specified layout
# for row_idx, img_id in enumerate(image_ids):
#     for col_idx, method in enumerate(methods):
#         # Building filename based on convention
#         if method == 'Image':
#             filename = f'Image_{img_id}.png'
            
#         # elif method == 'Mask':
#         #     filename = f'mask.png'
#         else:
#             filename = f'mask_{img_id}_{method}.png'

#         file_path = os.path.join(folder_path, filename)
        
#         if os.path.exists(file_path):
#             img = mpimg.imread(file_path)
#             axes[row_idx, col_idx].imshow(img, aspect='auto')
#         else:
#             # In case the image file does not exist, display a placeholder or text
#             axes[row_idx, col_idx].text(0.5, 0.5, 'Image not found', fontsize=12, ha='center', va='center', transform=axes[row_idx, col_idx].transAxes)

#         # Set titles for the top row
#         if row_idx == 0:
#             axes[row_idx, col_idx].set_title(method, fontsize=14, weight='bold')

# # Add row labels
# for ax, row in zip(axes[:,0], image_ids):
#     ax.set_ylabel(f'Image {row}', rotation=0, fontsize=14, labelpad=60, weight='bold', va='center')

# # Adjust overall layout to make sure everything fits perfectly
# plt.tight_layout()
# plt.show()



# ########################## ATTENTION MAPS FROM FOLDER ###########################
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# # Load attention maps
# inputs = np.load('Attention_Maps/inputs.npy')
# attn1 = np.load('Attention_Maps/attention_map_1.npy')
# attn2 = np.load('Attention_Maps/attention_map_2.npy')
# attn3 = np.load('Attention_Maps/attention_map_3.npy')

# output_dir = 'Attention_Maps/attention_maps.jpg'

# num_images = 4  # Assuming the first dimension is the batch size
# methods = ['MIP', '3D U-Net Activation Map', 'UNETR Attention Map', 'FCT Attention Map']
# fig, ax = plt.subplots(nrows=num_images, ncols=len(methods), figsize=(12, 12))  # Add one column for color bars

# for i in range(num_images):
#     input_image = inputs[i]
#     attn_map1 = attn1[i]
#     attn_map2 = attn2[i]
#     attn_map3 = attn3[i]

#     for j in range(len(methods)):
#         axes = ax[i, j]
#         im = axes.imshow(np.max(input_image, axis=2), cmap='gray', aspect='auto')
#         if j == 1:
#             im = axes.imshow(np.max(attn_map1, axis=2), cmap='jet', alpha=0.35, aspect='auto')
#         elif j == 2:
#             im = axes.imshow(np.max(attn_map2[0, 0], axis=2), cmap='jet', alpha=0.35, aspect='auto')
#         elif j == 3:
#             im = axes.imshow(np.max(attn_map3[0, 0], axis=2), cmap='jet', alpha=0.35, aspect='auto')
#         axes.axis('off')

#         # Add color bars in the last column of each row
#         if j == len(methods) - 1:
#             divider = make_axes_locatable(axes)
#             cax = divider.append_axes("right", size="5%", pad=0.05)
#             plt.colorbar(im, cax=cax)

#     # Titles for the top row
#     if i == 0:
#         for k, method in enumerate(methods):
#             ax[i, k].set_title(method, fontsize=10, weight='bold')

# plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust the outer dimensions if necessary
# fig.savefig(output_dir, dpi=300, bbox_inches='tight')  # Save the figure
# plt.show()


##################################################################################################

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# # Load attention maps
# inputs = np.load('Attention_Maps/inputs (4).npy')
# attn1 = np.load('Attention_Maps/attention_map_1 (2).npy')
# attn2 = np.load('Attention_Maps/attention_map_2 (2).npy')
# attn3 = np.load('Attention_Maps/attention_map_3 (2).npy')

# output_dir = 'Attention_Maps/attention_maps.jpg'

# num_images = 5  # Assuming the first dimension is the batch size
# methods = ['MIP', '3D U-Net', 'UNETR', 'FCT']
# # Create a dictionary to associate methods with attention maps
# attention_dict = {
#     'MIP': inputs,
#     '3D U-Net': attn1,
#     'UNETR': attn2,
#     'FCT': attn3
# }

# for j, method in enumerate(methods):
#     for i in range(num_images):
#         # Get the attention map for the current method and image
#         attn_map = attention_dict[method][i]
        
#         # Plot the attention map
#         if method == 'MIP':
#             plt.imshow(np.max(attn_map,axis=2), aspect='auto')
#             plt.axis('off')
#         elif method == '3D U-Net':
#             plt.imshow(np.max(attention_dict['MIP'][i], axis=2), cmap='gray', aspect='auto')
#             im = plt.imshow(np.max(attn_map,axis=2), cmap='jet', alpha=0.30, aspect='auto')
#             plt.axis('off')
#         else:
#             plt.imshow(np.max(attention_dict['MIP'][i], axis=2), cmap='gray', aspect='auto')
#             im = plt.imshow(np.max(attn_map[0, 0], axis=2), cmap='jet', alpha=0.30, aspect='auto')
#             plt.axis('off')
        
#         # Add color bars in the last column of each row, except for MIP
#         if method=='FCT':
#             divider = make_axes_locatable(plt.gca())
#             cax = divider.append_axes("right", size="5%", pad=0.05)
#             plt.colorbar(im, cax=cax)
        
#         # Save the attention map as an individual image
#         plt.savefig(f'Attention_Maps/Average/{method}_{i+1}.png', bbox_inches='tight', pad_inches=0)
#         plt.close()


###########################################################################################################

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np



# Define the plotting function
def plot_images(folder_path, methods, image_ids, y_labels, output_dir, size):
    # Set up the plot
    fig, axes = plt.subplots(nrows=len(image_ids), ncols=len(methods), figsize=(size[0], size[1]))
    # fig, axes = plt.subplots(nrows=len(image_ids), ncols=len(methods), figsize=(20, 20))
    

    axes = np.array(axes)  # Ensure axes is always 2-dimensional
    axes = axes.reshape(len(image_ids), -1)  # Reshape to 2D if necessary

    # Style adjustments for a more professional look
    plt.subplots_adjust(hspace=0.018, wspace=0.01)
    for ax in axes.flatten():
        ax.axis('off')

    # Load and display each image according to the specified layout
    for row_idx, img_id in enumerate(image_ids):
        
        for col_idx, method in enumerate(methods):
            # Building filename based on convention
            if method == 'Image':
                filename = f'image_{img_id}.png'
            elif method == 'Mask':
                filename = f'mask_{img_id}.png'
            else:
                filename = f'mask_{img_id}_{method}.png'
            
            file_path = os.path.join(folder_path, filename)

            if os.path.exists(file_path):
                img = mpimg.imread(file_path)
                axes[row_idx, col_idx].imshow(img, aspect='auto')
            else:
                axes[row_idx, col_idx].text(0.5, 0.5, 'Image not found', fontsize=11, ha='center', va='center', transform=axes[row_idx, col_idx].transAxes)

            # Set titles for the top row
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(method, fontsize=13)
       
    # Add row labels
    # for ax, label in zip(axes[:, 0], y_labels):
    #     ax.set_ylabel(label, rotation=0, fontsize=4000, labelpad=0, va='center')
        # ax.set_ylabel(label)
    for row_idx, label in enumerate(y_labels):
        # fig.text(0.10, 0.95 - (row_idx + 0.84) / (len(image_ids)+1), label, ha='center', va='center', rotation='horizontal', fontsize=13)
        y_position = 0.79 - (row_idx * 0.72 / (len(y_labels)-0.3))
        fig.text(0.10, y_position, label, ha='center', va='center', rotation='horizontal', fontsize=13)

    # plt.tight_layout()
    plt.show()
    fig.savefig(f'{output_dir[0]}.png', bbox_inches='tight', dpi=300)  

# List of plot parameters
plot_parameters = [
    # ('Full_images_96', ['Image', 'Mask', '3D U-Net', 'UNETR', 'FCT'], ['1', '6', '8'], ['a)', 'b)', 'c)'], ["Images/image_1"], [12,6]),
    # ('Full_images_256', ['Image', 'Mask', '3D U-Net', 'UNETR', 'FCT'], ['1', '6', '8'], ['a)', 'b)', 'c)'], ["Images/image_2"],[12,6]),
    #('Full_images_256/low_contrast', ['Image', 'Mask', '3D U-Net', 'UNETR', 'FCT'], ['8'], ['d)'], ["Images/image_3"], [12,2.5]),
    #('Full_images_256/noise', ['Image', 'Mask', '3D U-Net', 'UNETR', 'FCT'], ['8'], ['e)'], ["Images/image_4"], [12, 2.5]),
    # ('Generalization/Image_1', ['Image', '3D U-Net', 'UNETR', 'FCT'], ['10','Low_p6'], ['a)', 'b)'], ["Images/image_5"], [12, 5]),
    # ('Generalization/Image_1/low_contrast', ['Image', '3D U-Net', 'UNETR', 'FCT'], ['10', 'low_p6'], ['a-1)', 'b-1)'], ["Images/image_6"], [12, 5]),
    # ('Generalization/Image_1/pair_nuclei-golgi', ['Image', '3D U-Net', 'UNETR', 'FCT'], ['10', 'low_p6'], ['a-2)', 'b-2)'], ["Images/image_7"], [12, 5]),
    # ('Generalization/Image_1/error', ['Image', '3D U-Net', 'UNETR', 'FCT'], ['10_low_contrast', 'low_p6_low_contrast', '10_pair','low_p6_pair' ], ['a-1)', 'b-1)', 'a-2)', 'b-2)'], ["Images/image_18"], [8, 6]),
    #('Generalization/Image_2', ['Image', '3D U-Net', 'UNETR', 'FCT'], ['p6_icam', 'p6_icam_2'], ['c)', 'd)'], ["Images/image_8"],[12, 5]),
    # ('Generalization/Image_2/noise_nuclei', ['Image', '3D U-Net', 'UNETR', 'FCT'], ['icam', 'icam_2'], ['c-1)', 'd-1)'], ["Images/image_9"], [12, 5]),
    # ('Generalization/Image_2/noise_golgi', ['Image', '3D U-Net', 'UNETR', 'FCT'], ['icam', 'icam_2'], ['c-2)', 'd-2)'], ["Images/image_10"], [12, 5]),
    # ('Generalization/Image_2/error', ['Image', '3D U-Net', 'UNETR', 'FCT'], ['icam_nuclei', 'icam_2_nuclei','icam_golgi', 'icam_2_golgi'], ['c-1)', 'd-1)','c-2)', 'd-2)'], ["Images/image_19"], [8, 6]),
    #('Generalization/p6_avm', ['Image', '3D U-Net', 'UNETR', 'FCT'], ['p6_avm'], ['e)'], ["Images/image_11"], [12, 2.5]),
    ('Generalization/p6_avm/irregular_structures_1', ['Image', '3D U-Net', 'UNETR', 'FCT'], ['p6_avm', 'p6_avm_2'], ['e-1)', 'e-2)'], ["Images/image_12"], [10, 4]),
    # ("saved_images", ['Image', 'Mask', '3D U-Net','FCT'], ["0", '1', '2', '3', "4"], ['a)', 'b)', 'c)', "d)", "e)"], ["Images/image_13"], [9, 12])
    # ("Attention_maps/FCT/image_3", ['MIP', 'Block_1', 'Block_2', 'Block_3', 'Block_4', 'Block_5'], ["0"], ['a)'], ["Images/image_14"], [12, 3])
    # ("Attention_maps/Average", ['MIP', '3D U-Net', 'UNETR', 'FCT'], ["2", "3","4", "5"], ['f)','g)','h)','i)'], ["Images/image_17"], [9,15])
]

# Loop through each set of parameters and plot
for folder_path, methods, image_ids, y_labels, out, size in plot_parameters:
    plot_images(folder_path, methods, image_ids, y_labels, out, size)
