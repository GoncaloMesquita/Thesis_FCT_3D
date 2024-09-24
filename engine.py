
import torch
import numpy as np  
import torch.nn as nn
from utils import visualization_attn_maps_FCT_avg, visualization_attn_maps_UNETR_avg, visualization_attn_maps_3D_UNET, visualization_attn_maps_FCT_all_blocks, visualization_attn_maps_UNETR_all_blocks

def train(model, dataloader, optimizer, criterion, device, class_weight):
    
    model.train()
    running_loss = 0.0
    
    for inputs, targets in dataloader:
        
        inputs, targets = inputs.to(device, dtype=torch.float32), targets.to(device, dtype=torch.float32)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = 0
        for i in range(0,outputs.shape[1]):
            loss += criterion(outputs[:, i], targets[:, i]) * class_weight[i]
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device, class_weight, maps, name_model, output_dir):
    
    model.eval()
    
    running_loss = 0.0
    all_predictions = []
    all_targets = []
    attention = []
    inputs_rgb = []
    #Avaliar a situação para validation
    # with torch.no_grad():
    
    for batch_index, (inputs, targets) in enumerate(dataloader):
        
        inputs, targets = inputs.to(device, dtype=torch.float32), targets.to(device, dtype=torch.float32)
        
        outputs = model(inputs)
        loss = 0

        for i in range(0,outputs.shape[1]):
            loss += criterion(outputs[:, i], targets[:, i]) * class_weight[i]

        loss = criterion(outputs, targets)
        running_loss += loss.item()
        
        outputs =  torch.sigmoid(outputs)
        predictions = (outputs > 0.5).float().cpu().numpy()
        targets = targets.cpu().numpy()
        
        all_predictions.append(predictions)
        all_targets.append(targets)
        
        # if maps and batch_index % 2 != 0 and batch_index < 16:
        if maps and  batch_index < 16:

            if name_model == 'FCT':
                attention.append(visualization_attn_maps_FCT_all_blocks(model, inputs, f"Attention_Maps/FCT", batch_index+50))
                if batch_index == 15:

                    return inputs_rgb, attention

            if  name_model == 'UNETR':
                attention.append(visualization_attn_maps_UNETR_all_blocks(model, inputs, f"Attention_Maps/UNETR", batch_index+50))
                if batch_index == 15:

                    return inputs_rgb, attention

            if  name_model == '3D U-Net':
                atten1 = visualization_attn_maps_3D_UNET(model, inputs, outputs, f"Attention_Maps/3D_UNET", batch_index)
                attention.append(atten1)
                # inputs_rgb.append(input1)
                if batch_index == 15:

                    return inputs_rgb, attention

    val_loss = running_loss / len(dataloader)
    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)

    return val_loss, targets, predictions 


############################ Tentar observar em 2D Patches em Específico ############################################

# att_mat = model.vit.blocks[0].attn.att_mat # Chose the block to visualize the attention matrix
# att_mat_mean_head = torch.mean(att_mat, axis=1) # Average the attention matrix over the heads

# #att_mat_mean_head = att_mat[:,11]
# att_corner = att_mat_mean_head[0,0]   # Choose the attention matrix of the first patch (top left corner of the image)
# att_corner_reshaped = att_corner.reshape(6,6,6) # Reshape the attention matrix to be a 6x6x6 matrix
# att_corner_reshaped = att_corner_reshaped.unsqueeze(0).unsqueeze(0)
# heatmap_corner = torch.nn.functional.interpolate(att_corner_reshaped, scale_factor=(96 // 6) , mode='trilinear', align_corners=False) #Do interpolation to match the input size
# #Visualização
# inputs_np = inputs[0].permute(1, 2, 3, 0).cpu().numpy()
# image_rgb = np.zeros((inputs.shape[2], inputs.shape[3], inputs.shape[4], 3), dtype=np.uint8)
# image_rgb[:,:,:,0] = inputs_np[:,:,:,0]*255
# image_rgb[:,:,:,1] = inputs_np[:,:,:,1]*255


# heatmap_corner_norm = (heatmap_corner - heatmap_corner.min()) / (heatmap_corner.max() - heatmap_corner.min()).numpy()
# heatmap_corner_norm  = heatmap_corner_norm.numpy()
# fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# # Plot the original image
# ax[0].imshow(image_rgb[:,:,35,:])
# ax[0].set_title("Original Image")

# # Plot the original image with the attention heatmap overlay
# ax[1].imshow(image_rgb[:,:,35,:])
# ax[1].imshow(heatmap_corner_norm[0,0,:,:,35], cmap='jet', alpha=0.35)  # Overlay with transparency
# ax[1].set_title("Image with Attention Heatmap")

# plt.tight_layout()

##################################################  Tentar observar em 3D Patches em Específico ############################################

# num_patch = 23
# num_block = 0
# shape = (6,6,6)
# scale = 96 // 6
# patch_size = (16,16,16)

# output_dir = f'Results/Normal/Results_{args.model}_{args.patch_size[0]}/attention_maps'
# os.makedirs(output_dir, exist_ok=True)

# att_mat = model.vit.blocks[num_block].attn.att_mat # Chose the block to visualize the attention matrix
# att_mat_mean_head = torch.mean(att_mat, axis=1) # Average the attention matrix over the heads
# att_corner = att_mat_mean_head[0,num_patch]   # Choose the attention matrix of the first patch (top left corner of the image)

# #att_corner = attention_avg.sum(dim=1)
# att_corner_reshaped = att_corner.reshape(shape) # Reshape the attention matrix to be a 6x6x6 matrix
# att_corner_reshaped = att_corner_reshaped.unsqueeze(0).unsqueeze(0)
# heatmap_corner = torch.nn.functional.interpolate(att_corner_reshaped, scale_factor=(scale) , mode='trilinear', align_corners=True) #Do interpolation to match the input size

# z = num_patch // (shape[1] * shape[2])
# # Calculate the remainder to find the position within the z-slice
# remainder_z = num_patch % (shape[1] * shape[2])
# # Calculate the row (y-coordinate) within the z-slice
# y = remainder_z % shape[2]
# # Calculate the column (x-coordinate) within the z-slice
# x = remainder_z // shape[2] 

# # Create a 3D rectangle to highlight the patch
# rectangle = np.zeros((heatmap_corner.shape[2], heatmap_corner.shape[3], heatmap_corner.shape[4]), dtype=bool)
# rectangle[x*scale:x*scale+patch_size[0], y*scale:y*scale+patch_size[1], z*scale:z*scale+patch_size[2]] = True

# # Add the rectangle to the heatmap
# heatmap_corner[0, 0, rectangle] = 1.0


# #Visualização
# inputs_np = inputs[0].permute(1, 2, 3, 0).cpu().numpy()
# image_rgb = np.zeros((inputs.shape[2], inputs.shape[3], inputs.shape[4], 3), dtype=np.uint8)
# image_rgb[:,:,:,0] = inputs_np[:,:,:,0]*255
# image_rgb[:,:,:,1] = inputs_np[:,:,:,1]*255

# r - heatmap_corner.min()) / (heatmap_corner.max() - heatmap_corner.min()).numpy()
# heatmap_corner_norm  = heatmap_corner.numpy()

# vmin, vmax = np.percentile(heatmap_corner_norm, 30), np.percentile(heatmap_corner_norm, 98)
# heatmap_clipped = np.clip(heatmap_corner_norm, vmin, vmax)
# heatmap_clipped_normalized = (heatmap_clipped - vmin) / (vmax - vmin)
# cmap = plt.get_cmap("jet") 

# heatmap_rgb = cmap(heatmap_clipped_normalized)
# # heatmap_rgb = (heatmap_rgb[..., :3]*255).astype(np.uint8)

# alpha_scale = np.interp(heatmap_corner_norm, (heatmap_corner_norm.min(), heatmap_corner_norm.max()), (0.1, 1))  # Adjust transparency range as needed
# heatmap_rgb[..., 3] = alpha_scale  # Apply the scaled alpha values

# # Convert to RGB, considering alpha for blending with white background (or any other chosen background)
# background_color = np.array([1, 1, 1])  # White
# heatmap_rgb_with_alpha = heatmap_rgb[..., :3] * heatmap_rgb[..., 3, np.newaxis] + background_color * (1 - heatmap_rgb[..., 3, np.newaxis])
# heatmap_rgb_with_alpha = (heatmap_rgb_with_alpha * 255).astype(np.uint8)

# import tifffile
# plt.imshow(heatmap_rgb_with_alpha[0,0,:,:,35])
# plt.show()
# tifffile.imwrite(f'Results/Results_UNETR_96/attention_corner.tif', heatmap_rgb[0,0])



##########################################  Get the attention map of which patches were the most important ######################################




# import matplotlib.pyplot as plt
# import numpy as np
# import torch

# # Assuming 'model' is your model and 'inputs' is your input tensor
# fig, ax = plt.subplots(4, 3, figsize=(14, 18))  # Adjusted figure size for clarity

# # Define the indices of blocks you want to visualize
# block_indices = [0, 3, 7, 11]  # Example: Choose blocks at intervals for diversity
# fig.subplots_adjust(hspace=0.01, wspace=0)
# for idx, block_idx in enumerate(block_indices):
#     # Extract attention matrix for the specified block
#     att_mat = model.vit.blocks[block_idx].attn.att_mat
#     att_mat_mean_head = torch.mean(att_mat, axis=1)  # Average over the heads
#     att_mat_mean_head = att_mat_mean_head.sum(dim=1)
    
#     # Process and interpolate the attention heatmap
#     att_corner = att_mat_mean_head[0]  # Assuming focusing on the first patch
#     # att_corner_reshaped = att_corner.reshape(6, 6, 6)
#     att_corner_reshaped = att_corner.reshape(16,16,4)
#     # heatmap_corner = torch.nn.functional.interpolate(att_corner_reshaped.unsqueeze(0).unsqueeze(0), scale_factor=(96 // 6), mode='trilinear', align_corners=False)
#     heatmap_corner = torch.nn.functional.interpolate(att_corner_reshaped.unsqueeze(0).unsqueeze(0), scale_factor=(256 // 16), mode='trilinear', align_corners=False)
    
#     # Normalize the heatmap for visualization
#     heatmap_corner_norm = (heatmap_corner - heatmap_corner.min()) / (heatmap_corner.max() - heatmap_corner.min())
#     heatmap_corner_norm = heatmap_corner_norm.numpy()
    
#     # Prepare the original image for display
#     inputs_np = inputs[0].permute(1, 2, 3, 0).cpu().numpy()
#     image_rgb = np.zeros((inputs.shape[2], inputs.shape[3], inputs.shape[4], 3), dtype=np.uint8)
#     image_rgb[:, :, :, 0] = inputs_np[:, :, :, 0] * 255
#     image_rgb[:, :, :, 1] = inputs_np[:, :, :, 1] * 255  # Adjust channels as necessary
    
#     # Plot original image
#     ax[idx, 0].imshow(image_rgb[:, :, 35, :])  # Adjust slice index as needed
#     ax[idx, 0].set_title(f"Block {block_idx} - Original Image", fontsize=11)
#     ax[idx, 0].axis('off')  # Hide axes for a cleaner look
    
#     # Plot heatmap overlay
#     ax[idx, 1].imshow(image_rgb[:, :, 35, :])
#     ax[idx, 1].imshow(heatmap_corner_norm[0, 0, :, :, 35], cmap='jet', alpha=0.15)  # Adjusted alpha for visibility
#     ax[idx, 1].set_title(f"Block {block_idx} - Attention heatmap and Original Image", fontsize=11)
#     ax[idx, 1].axis('off')
    
#     heatmap_color = ax[idx, 2].imshow(heatmap_corner_norm[0, 0, :, :, 35], cmap='jet', alpha=1)  # Adjusted alpha for visibility
#     ax[idx,2].set_title(f"Block {block_idx} - Attention heatmap slice 35", fontsize=11)
#     ax[idx,2].axis('off')
#     cbar = fig.colorbar(heatmap_color, ax=ax[idx,2], fraction=0.036, pad=0.004)
#     cbar.set_label('Attention Intensity', fontsize=11)
    

# plt.subplots_adjust(wspace=0.01, hspace=0.2)
# plt.show()



###############################################  Get the average from all the blocks #############################################

# import matplotlib.pyplot as plt

# # Assuming 'model' is your model and 'inputs' is your input tensor
# fig, ax = plt.subplots(1, 3, figsize=(12, 6))

# # Define the indices of blocks you want to visualize
# block_indices = range(len(model.vit.blocks))

# # Initialize variables to store the average attention heatmap
# avg_heatmap = None

# for block_idx in block_indices:
#     # Extract attention matrix for the specified block
#     att_mat = model.vit.blocks[block_idx].attn.att_mat
#     att_mat_mean_head = torch.mean(att_mat, axis=1)  # Average over the heads
#     att_mat_mean_head = att_mat_mean_head.sum(dim=1)
    
#     # Process and interpolate the attention heatmap
#     att_corner = att_mat_mean_head[0]  
#     # att_corner_reshaped = att_corner.reshape(6, 6, 6)
#     att_corner_reshaped = att_corner.reshape(16,16,4)
    
#     # heatmap_corner = torch.nn.functional.interpolate(att_corner_reshaped.unsqueeze(0).unsqueeze(0), scale_factor=(96 // 6), mode='trilinear', align_corners=False)
#     heatmap_corner = torch.nn.functional.interpolate(att_corner_reshaped.unsqueeze(0).unsqueeze(0), scale_factor=(256 // 16), mode='trilinear', align_corners=False)
    
#     # Normalize the heatmap for visualization
#     heatmap_corner_norm = (heatmap_corner - heatmap_corner.min()) / (heatmap_corner.max() - heatmap_corner.min())
#     heatmap_corner_norm = heatmap_corner_norm.numpy()
    
#     # Add the current heatmap to the average
#     if avg_heatmap is None:
#         avg_heatmap = heatmap_corner_norm
#     else:
#         avg_heatmap += heatmap_corner_norm

# # Calculate the average heatmap
# avg_heatmap /= len(block_indices)

# # Prepare the original image for display
# inputs_np = inputs[0].permute(1, 2, 3, 0).cpu().numpy()
# image_rgb = np.zeros((inputs.shape[2], inputs.shape[3], inputs.shape[4], 3), dtype=np.uint8)
# image_rgb[:, :, :, 0] = inputs_np[:, :, :, 0] * 255
# image_rgb[:, :, :, 1] = inputs_np[:, :, :, 1] * 255  # Adjust channels as necessary

# # Plot original image
# ax[0].imshow(image_rgb[:, :, 35, :])  # Adjust slice index as needed
# ax[0].set_title("Original Image", fontsize=11)
# ax[0].axis('off')  # Hide axes for a cleaner look

# # Plot average heatmap overlay
# ax[1].imshow(image_rgb[:, :, 35, :])
# ax[1].imshow(avg_heatmap[0, 0, :, :, 35], cmap='jet', alpha=0.5)  # Adjusted alpha for visibility
# ax[1].set_title("Average Attention Heatmap", fontsize=11)
# ax[1].axis('off')

# heatmap_color = ax[2].imshow(avg_heatmap[0, 0, :, :, 35], cmap='jet', alpha=1)  # Adjusted alpha for visibility
# ax[2].set_title("Nuclei Activation Heatmap slice 35", fontsize=11)
# ax[2].axis('off')
# cbar = fig.colorbar(heatmap_color, ax=ax[2], fraction=0.036, pad=0.004)
# cbar.set_label('Activation Intensity', fontsize=11)

# plt.tight_layout()
# plt.show()



############################################### 3DUnet   #############################################


    # gradient = torch.ones_like(outputs[:, 0])
    # outputs[:, 0].backward(gradient, retain_graph=True)

    # gradients = model.get_activations_gradient()

    # # pool the gradients across the channels
    # pooled_gradients = torch.mean(gradients, dim=[0, 2, 3, 4])

    # # get the activations of the last convolutional layer
    # activations = model.get_activations(inputs).detach()

    # # weight the channels by corresponding gradients
    # for i in range(gradients.shape[1]):
    #     activations[:, i, :, :] *= pooled_gradients[i]
        
    # # average the channels of the activations
    # heatmap = torch.mean(activations, dim=1).squeeze()
    
    
    # # Convert 3D heatmap to 2D by averaging across slices
    # heatmap_int = torch.nn.functional.interpolate(heatmap.unsqueeze(0).unsqueeze(0), scale_factor=(96 // 12), mode='trilinear', align_corners=False)
    # heatmap_int = torch.nn.functional.interpolate(heatmap.unsqueeze(0).unsqueeze(0), scale_factor=(256 // 16), mode='trilinear', align_corners=False)
    
    # heatmap_corner_norm = (heatmap_int - heatmap_int.min()) / (heatmap_int.max() - heatmap_int.min())
    # heatmap_corner_norm = heatmap_corner_norm.numpy()
    # inputs_np = inputs[0].permute(1, 2, 3, 0).cpu().numpy()
    # image_rgb = np.zeros((inputs.shape[2], inputs.shape[3], inputs.shape[4], 3), dtype=np.uint8)
    # image_rgb[:, :, :, 0] = inputs_np[:, :, :, 0] * 255
    # image_rgb[:, :, :, 1] = inputs_np[:, :, :, 1] * 255  # Adjust channels as necessary

    # fig, ax = plt.subplots(1, 3, figsize=(12, 6))

    # # Plot original image
    # ax[0].imshow(image_rgb[:, :, 35, :])  # Adjust slice index as needed
    # ax[0].set_title("Original Image Slice 35", fontsize=11)
    # ax[0].axis('off')  # Hide axes for a cleaner look

    # # Plot average heatmap overlay
    # ax[1].imshow(image_rgb[:, :, 35, :])
    # ax[1].imshow(heatmap_corner_norm[0, 0, :, :, 35], cmap='jet', alpha=0.15)  # Adjusted alpha for visibility
    # ax[1].set_title("Nuclei Activation Heatmap and Original Image slice 35", fontsize=11)
    # ax[1].axis('off')

    #     # Plot average heatmap overlay
    # heatmap_color=ax[2].imshow(heatmap_corner_norm[0, 0, :, :, 35], cmap='jet', alpha=1)  # Adjusted alpha for visibility
    # ax[2].set_title("Nuclei Activation Heatmap slice 35", fontsize=11)
    # ax[2].axis('off')
    # cbar = fig.colorbar(heatmap_color, ax=ax[2], fraction=0.036, pad=0.004)
    # cbar.set_label('Activation Intensity', fontsize=11)
    # plt.tight_layout()
    # plt.show()
    
    ############################################  FCT first attention block #########################################
    

# import matplotlib.pyplot as plt
# import numpy as np
# import torch

# # Assuming 'model' is your model and 'inputs' is your input tensor
# fig, ax = plt.subplots(1, 2, figsize=(20, 24))  # Adjusted figure size for clarity

# # Define the indices of blocks you want to visualize

#     # Extract attention matrix for the specified block
# att_mat = model.block_1.trans.attention_output.att_mat
# # att_mat_mean_head = torch.mean(att_mat, axis=1)  # Average over the heads
# att_mat_mip = att_mat[0].sum(dim=0)
# attn_mip_reshape = att_mat_mip.reshape(16,16,16)
# heatmap = torch.nn.functional.interpolate(attn_mip_reshape.unsqueeze(0).unsqueeze(0), scale_factor=(96 // 16), mode='trilinear', align_corners=False)

# # Normalize the heatmap for visualization
# heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
# heatmap_norm = heatmap_norm.numpy()

# # Prepare the original image for display
# inputs_np = inputs[0].permute(1, 2, 3, 0).cpu().numpy()
# image_rgb = np.zeros((inputs.shape[2], inputs.shape[3], inputs.shape[4], 3), dtype=np.uint8)
# image_rgb[:, :, :, 0] = inputs_np[:, :, :, 0] * 255
# image_rgb[:, :, :, 1] = inputs_np[:, :, :, 1] * 255  # Adjust channels as necessary

# # Plot original image
# ax[ 0].imshow(image_rgb[:, :, 35, :])  # Adjust slice index as needed
# ax[ 0].set_title(f"Block 0- Original Image", fontsize=14)
# ax[ 0].axis('off')  # Hide axes for a cleaner look

# # Plot heatmap overlay
# ax[ 1].imshow(image_rgb[:, :, 35, :])
# ax[ 1].imshow(heatmap_norm[0, 0, :, :, 35], cmap='jet', alpha=0.5)  # Adjusted alpha for visibility
# ax[ 1].set_title(f"Block 0 - Attention Heatmap", fontsize=14)
# ax[ 1].axis('off')

# plt.tight_layout()
# plt.show()



##################################### FCT Get most important patches in all blcks #######################################


# import matplotlib.pyplot as plt
# import numpy as np
# import torch

# # Assuming 'model' is your model and 'inputs' is your input tensor
# fig, ax = plt.subplots(3, 5, figsize=(20, 12))  # Transpose the subplot grid for horizontal blocks and vertical images
# plt.subplots_adjust(hspace=0.1, wspace=0.1)  # Tighten the spacing for a more compact layout

# blocks = [
#     model.block_1.trans.attention_output.att_mat, 
#     model.block_2.trans.attention_output.att_mat,
#     model.block_3.trans.attention_output.att_mat,
#     model.block_4.trans.attention_output.att_mat,
#     model.block_5.trans.attention_output.att_mat
# ]

# patches = [(32,32,16), (32,32,8), (32,32,8), (16,16,4), (8,8,2)]
# for m, (block, patch_size) in enumerate(zip(blocks, patches)):
#     att_mat = block
#     att_mat_mip = att_mat[0].sum(dim=0)
#     attn_mip_reshape = att_mat_mip.reshape(patch_size)
#     heatmap = torch.nn.functional.interpolate(
#         attn_mip_reshape.unsqueeze(0).unsqueeze(0), 
#         scale_factor=(256 // patch_size[0], 256 // patch_size[1], 64 // patch_size[2]), 
#         mode='trilinear'
#     )

#     # Normalize the heatmap for visualization
#     heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
#     heatmap_norm = heatmap_norm.numpy()

#     # Prepare the original image for display
#     inputs_np = inputs[0].permute(1, 2, 3, 0).cpu().numpy()
#     image_rgb = np.zeros((inputs.shape[2], inputs.shape[3], inputs.shape[4], 3), dtype=np.uint8)
#     image_rgb[..., 0] = inputs_np[..., 0] * 255
#     image_rgb[..., 1] = inputs_np[..., 1] * 255

#     # Plot original image
#     ax[0, m].imshow(np.max(image_rgb, axis=2))
#     ax[0, m].set_title(f"Block {m+1} Original Image", fontsize=10)
#     ax[0, m].axis('off')

#     # Overlay Heatmap on Image
#     heat_mask = (np.max(heatmap_norm[0, 0], axis=2) > 0.05)  # Thresholding
#     image_highlighted = np.max(image_rgb, axis=2) * heat_mask[..., np.newaxis]
#     ax[1, m].imshow(image_highlighted)
#     ax[1, m].set_title(f"Block {m+1} Attention + Image", fontsize=10)
#     ax[1, m].axis('off')

#     # Plotting heatmap
#     heatmap_plotted = ax[2, m].imshow(np.max(heatmap_norm[0, 0], axis=2), cmap='jet')
#     ax[2, m].set_title(f"Block {m+1} Attention Heatmap", fontsize=10)
#     ax[2, m].axis('off')

#     # Add colorbars only to the last heatmap in each row for clarity
#     if m == 4:
#         cbar = fig.colorbar(heatmap_plotted, ax=ax[2, m], fraction=0.046, pad=0.04)
#         cbar.set_label('Attention Intensity', fontsize=10)

# plt.tight_layout()
# plt.show()


########################################## average FCT all blocks #########################################


# import matplotlib.pyplot as plt

# Assuming 'model' is your model and 'inputs' is your input tensor
# fig, ax = plt.subplots(1, 3, figsize=(12, 6))

# # Define the indices of blocks you want to visualize
# patches = [(32,32,16), (16,16,8), (8,8,4), (4,4,2), (2,2,1)]

# # Initialize variables to store the average attention heatmap
# avg_heatmap = None
# blocks = [
#     model.block_1.trans.attention_output.att_mat, 
#     model.block_2.trans.attention_output.att_mat,
#     model.block_3.trans.attention_output.att_mat,
#     model.block_4.trans.attention_output.att_mat,
#     model.block_5.trans.attention_output.att_mat
# ]
# for m, (block, patch_size) in enumerate(zip(blocks, patches)):
#     # Extract attention matrix for the specified block
#     att_mat = block
#     #att_mat = (att_mat - att_mat.min()) / (att_mat.max() - att_mat.min())
#     print(att_mat)
#     att_corner = att_mat[0].sum(dim=0)
#     print(m)
#     # Process and interpolate the attention heatmap
#     # att_corner_reshaped = att_corner.reshape(6, 6, 6)
#     att_corner_reshaped = att_corner.reshape(patch_size)
    
#     # heatmap_corner = torch.nn.functional.interpolate(att_corner_reshaped.unsqueeze(0).unsqueeze(0), scale_factor=(96 // 6), mode='trilinear', align_corners=False)
#     heatmap_corner = torch.nn.functional.interpolate(att_corner_reshaped.unsqueeze(0).unsqueeze(0), scale_factor=(256 // patch_size[0]), mode='trilinear', align_corners=False)
    
#     # Normalize the heatmap for visualization
#     heatmap_corner_norm = (heatmap_corner - heatmap_corner.min()) / (heatmap_corner.max() - heatmap_corner.min())
#     heatmap_corner_norm = heatmap_corner_norm.numpy()
    
#     # Add the current heatmap to the average
#     if avg_heatmap is None:
#         avg_heatmap = heatmap_corner_norm
#     else:
#         avg_heatmap += heatmap_corner_norm

# # Calculate the average heatmap
# avg_heatmap = (avg_heatmap - avg_heatmap.min()) / (avg_heatmap.max() - avg_heatmap.min())
# avg_heatmap /= len(blocks)

# # Prepare the original image for display
# inputs_np = inputs[0].permute(1, 2, 3, 0).cpu().numpy()
# image_rgb = np.zeros((inputs.shape[2], inputs.shape[3], inputs.shape[4], 3), dtype=np.uint8)
# image_rgb[:, :, :, 0] = inputs_np[:, :, :, 0] * 255
# image_rgb[:, :, :, 1] = inputs_np[:, :, :, 1] * 255  # Adjust channels as necessary

# # Plot original image
# ax[0].imshow(image_rgb[:, :, 35, :])  # Adjust slice index as needed
# ax[0].set_title("Original Image", fontsize=11)
# ax[0].axis('off')  # Hide axes for a cleaner look

# # Plot average heatmap overlay
# ax[1].imshow(image_rgb[:, :, 35, :])
# ax[1].imshow(avg_heatmap[0, 0, :, :, 35], cmap='jet', alpha=0.25)  # Adjusted alpha for visibility
# ax[1].set_title("Average Attention Heatmap", fontsize=11)
# ax[1].axis('off')




