
import torch
import numpy as np
from torchvision.transforms import Compose
from monai.transforms import (
    RandFlipd,
    RandRotate90d,
)
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import os


class EarlyStopping:
    
    def __init__(self, patience, patch_size, model_name, output_dir, verbose=True, delta=0):
        
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.patch = patch_size
        self.output = output_dir
        self.model = model_name

    def __call__(self, val_loss, model, fold, epoch):

        self.fold = fold
        self.epoch = epoch

        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        if self.verbose:
            print("Validation loss improved. Saving the model...")
        torch.save(model.state_dict(), f"{self.output}/{self.model}_{self.patch}_{self.fold}_best.pth")
        return f"{self.output}/{self.model}_{self.patch}_{self.fold}_best.pth"


def reconstruct_image_from_patches(patches, indices, original_shape, pad):
    padding_x, padding_y, padding_z = pad
    reconstructed_image = np.zeros(original_shape)
    reconstructed_image_nuclei = np.zeros(original_shape)
    reconstructed_image_golgi = np.zeros(original_shape)
    reconstructed_image_back = np.zeros(original_shape)
    patches = patches.transpose(0, 2, 3, 4, 1)
    for patch, (i, j, k) in zip(patches, indices):

        reconstructed_image_back[i:i + patch.shape[0], j:j + patch.shape[1], k:k + patch.shape[2], :] += (patch[:, :, :, :] == 0).astype(int)
        reconstructed_image_nuclei[i:i + patch.shape[0], j:j + patch.shape[1], k:k + patch.shape[2], 1] += (patch[:, :, :, 1] > 0).astype(int)
        reconstructed_image_golgi[i:i + patch.shape[0], j:j + patch.shape[1], k:k + patch.shape[2], 0] += (patch[:, :, :, 0] > 0).astype(int)   

    reconstructed_image[:,:,:,0] = (reconstructed_image_golgi[:,:,:,0] > reconstructed_image_back[:,:,:,0]).astype(int)
    reconstructed_image[:,:,:,1] = (reconstructed_image_nuclei[:,:,:,1] > reconstructed_image_back[:,:,:,1]).astype(int)
    
    image = reconstructed_image[
    padding_x:,
    padding_y:,
    padding_z:,]

    return image


transformations_train = Compose(
    [
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.3,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.3,
        ),
        RandRotate90d(
          keys=["image", "label"],
            prob=0.3,
            max_k=1,
            spatial_axes=(0,1),
        )
    ]
)



def visualization_attn_maps_FCT_all_blocks(model, inputs, file_path, image_id):
    # Create a folder with file path and image id
    folder_path = os.path.join(file_path, f'image_{image_id}')
    os.makedirs(folder_path, exist_ok=True)
    
    blocks = [
        model.block_1.trans.attention_output.att_mat, 
        model.block_2.trans.attention_output.att_mat,
        model.block_3.trans.attention_output.att_mat,
        model.block_4.trans.attention_output.att_mat,
        model.block_5.trans.attention_output.att_mat
    ]

    patches = [(32,32,16), (32,32,8), (32,32,8), (16,16,4), (8,8,2)]
    # fig, ax = plt.subplots(1, 1 + len(blocks), figsize=(20, 6))  
    
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    
    inputs_np = inputs[0].permute(1, 2, 3, 0).cpu().numpy()
    image_rgb = np.zeros((inputs.shape[2], inputs.shape[3], inputs.shape[4], 3), dtype=np.uint8)
    image_rgb[..., 0] = inputs_np[..., 0] * 255
    image_rgb[..., 1] = inputs_np[..., 1] * 255
   
    plt.imshow(np.max(image_rgb, axis=2))
    plt.axis('off')
    plt.savefig(os.path.join(folder_path, f'MIP_{image_id}'), bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # ax[0].imshow(np.max(image_rgb, axis=2))
    # ax[0].set_title(f"MIP Image", fontsize=10, weight='bold')
    # ax[0].axis('off')
    
    treshold = [0.15, 0.015, 0.015, 0.03, 0.175]
    
    for m, (block, patch_size) in enumerate(zip(blocks, patches)):
        
        att_map = block.detach()
        att_mat_mip = att_map[0].sum(dim=0)
        
        attn_mip_reshape = att_mat_mip.reshape(patch_size)
        
        heatmap = torch.nn.functional.interpolate(
            attn_mip_reshape.unsqueeze(0).unsqueeze(0), 
            scale_factor=(256 // patch_size[0], 256 // patch_size[1], 64 // patch_size[2]), 
            mode='trilinear'
        )

        # Normalize the heatmap for visualization
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        heatmap_norm = heatmap_norm.numpy()

        # Overlay Heatmap on Image
        heat_mask = (np.max(heatmap_norm[0, 0], axis=2) > treshold[m])  # Thresholding
        image_highlighted = np.max(image_rgb, axis=2) * heat_mask[..., np.newaxis]
        # ax[m+1].imshow(image_highlighted)
        # ax[m+1].set_title(f"Block {m+1}: MIP with Attention", fontsize=10, weight='bold')
        # ax[m+1].axis('off')

        plt.imshow(image_highlighted)
        plt.axis('off')
        plt.savefig(os.path.join(folder_path, f'Block_{m+1}.png'), bbox_inches='tight', pad_inches=0)
        plt.close()


    # plt.tight_layout()
    # plt.show()
    # fig.savefig(file_path)
    
    return


def visualization_attn_maps_UNETR_all_blocks(model, inputs, file_path, image_id):

    # fig, ax = plt.subplots(4, 3, figsize=(14, 18)) 
    folder_path = os.path.join(file_path, f'image_{image_id}')
    os.makedirs(folder_path, exist_ok=True)
    num_blocks = len(model.vit.blocks)
    block_indices = []  
    
    # fig.subplots_adjust(hspace=0.01, wspace=0)
    
    inputs_np = inputs[0].permute(1, 2, 3, 0).cpu().numpy()
    image_rgb = np.zeros((inputs.shape[2], inputs.shape[3], inputs.shape[4], 3), dtype=np.uint8)
    image_rgb[:, :, :, 0] = inputs_np[:, :, :, 0] * 255
    image_rgb[:, :, :, 1] = inputs_np[:, :, :, 1] * 255  # Adjust channels as necessary
    
    plt.imshow(np.max(image_rgb, axis=2))
    plt.axis('off')
    plt.savefig(os.path.join(folder_path, f'MIP_{image_id}'), bbox_inches='tight', pad_inches=0)
    plt.close()
    
    for  block_idx in range(num_blocks):
        
        att_mat = model.vit.blocks[block_idx].attn.att_mat
        att_mat_mean_head = torch.mean(att_mat, axis=1)  # Average over the heads
        att_mat_mean_head = att_mat_mean_head.sum(dim=1)
        
        att_corner = att_mat_mean_head[0] 
       
        att_corner_reshaped = att_corner.reshape(16,16,4)
        # heatmap_corner = torch.nn.functional.interpolate(att_corner_reshaped.unsqueeze(0).unsqueeze(0), scale_factor=(96 // 6), mode='trilinear', align_corners=False)
        heatmap_corner = torch.nn.functional.interpolate(att_corner_reshaped.unsqueeze(0).unsqueeze(0), scale_factor=(256 // 16), mode='trilinear', align_corners=False)
        
        # Normalize the heatmap for visualization
        heatmap_norm = (heatmap_corner - heatmap_corner.min()) / (heatmap_corner.max() - heatmap_corner.min())
        heatmap_norm = heatmap_norm.numpy()
        
        # Prepare the original image for display
        
        # Overlay Heatmap on Image
        heat_mask = (np.max(heatmap_norm[0, 0], axis=2) > 0.1)  # Thresholding
        image_highlighted = np.max(image_rgb, axis=2) * heat_mask[..., np.newaxis]
        # ax[m+1].imshow(image_highlighted)
        # ax[m+1].set_title(f"Block {m+1}: MIP with Attention", fontsize=10, weight='bold')
        # ax[m+1].axis('off')

        plt.imshow(image_highlighted)
        plt.axis('off')
        plt.savefig(os.path.join(folder_path, f'Block_{block_idx+1}.png'), bbox_inches='tight', pad_inches=0)
        plt.close()

        
        # Plot original image
        # ax[idx, 0].imshow(np.max(image_rgb, axis=2))  # Adjust slice index as needed
        # ax[idx, 0].set_title(f"Block {block_idx} MIP Image", fontsize=11)
        # ax[idx, 0].axis('off')  # Hide axes for a cleaner look
        
        # # Plot heatmap overlay
        # ax[idx, 1].imshow(np.max(image_rgb, axis=2))
        # ax[idx, 1].imshow(np.max(heatmap_norm[0, 0], axis=2), cmap='jet', alpha=0.15)  # Adjusted alpha for visibility
        # ax[idx, 1].set_title(f"Block {block_idx} - Attention Map and MIP Image", fontsize=11)
        # ax[idx, 1].axis('off')
        
        # heatmap_color = ax[idx, 2].imshow(np.max(heatmap_norm[0, 0], axis=2), cmap='jet', alpha=1)  # Adjusted alpha for visibility
        # ax[idx,2].set_title(f"Block {block_idx} - Attention Map", fontsize=11)
        # ax[idx,2].axis('off')
        # cbar = fig.colorbar(heatmap_color, ax=ax[idx,2], fraction=0.036, pad=0.004)
        # cbar.set_label('Attention Intensity', fontsize=11)
        

    # plt.subplots_adjust(wspace=0.01, hspace=0.2)
    # plt.show()
    # fig.savefig(file_path)
    
    return heatmap_norm
    
    

def visualization_attn_maps_FCT_avg(model, inputs, file_path):
    
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    
    patches = [(32, 32, 16), (32, 32, 8), (32, 32, 8), (16, 16, 4), (8, 8, 2)]
    
    avg_heatmap = None
    
    blocks = [
        model.block_1.trans.attention_output.att_mat, 
        model.block_2.trans.attention_output.att_mat,
        model.block_3.trans.attention_output.att_mat,
        model.block_4.trans.attention_output.att_mat,
    ]
    
    for block, patch_size in zip(blocks, patches):
        att_mat = block
        att_mat = F.softmax(att_mat, dim=-1)
        att_corner = att_mat[0].sum(dim=0)
        
        att_corner_reshaped = att_corner.reshape(patch_size)
        heatmap_corner = torch.nn.functional.interpolate(att_corner_reshaped.unsqueeze(0).unsqueeze(0), scale_factor=(256 // patch_size[0], 256 // patch_size[1],64 // patch_size[2] ), mode='trilinear')
        
        heatmap_corner_norm = (heatmap_corner - heatmap_corner.min()) / (heatmap_corner.max() - heatmap_corner.min())
        heatmap_corner_norm = heatmap_corner_norm.numpy()
        
        if avg_heatmap is None:
            avg_heatmap = heatmap_corner_norm
        else:
            avg_heatmap += heatmap_corner_norm
    avg_heatmap /= len(blocks)
    avg_heatmap = (avg_heatmap - avg_heatmap.min()) / (avg_heatmap.max() - avg_heatmap.min())
    
    
    inputs_np = inputs[0].permute(1, 2, 3, 0).cpu().numpy()
    image_rgb = np.zeros((inputs.shape[2], inputs.shape[3], inputs.shape[4], 3), dtype=np.uint8)
    image_rgb[:, :, :, 0] = inputs_np[:, :, :, 0] * 255
    image_rgb[:, :, :, 1] = inputs_np[:, :, :, 1] * 255
    
    # ax[0].imshow(np.max(image_rgb,axis=2))
    # ax[0].set_title("MIP Image", fontsize=14, weight='bold')
    # ax[0].axis('off')
    
    # ax[1].imshow(np.max(image_rgb,axis=2))
    # ax[1].imshow(np.max(avg_heatmap[0,0], axis=2), cmap='jet', alpha=0.25)
    # ax[1].set_title("Average Attention Heatmap and MIP Image", fontsize=14, weight='bold')
    # ax[1].axis('off')
    
    # heatmap_color = ax[2].imshow(np.max(avg_heatmap[0,0], axis=2), cmap='jet')
    # ax[2].set_title("Average Attention Heatmap", fontsize=14, weight='bold')
    # ax[2].axis('off')
    
    # cbar = fig.colorbar(heatmap_color, ax=ax[2], fraction=0.046, pad=0.04)
    # cbar.set_label('Attention Intensity', fontsize=14)
    
    # plt.tight_layout()
    # plt.savefig(file_path)
    # plt.show()
    
    return avg_heatmap


def visualization_attn_maps_UNETR_avg(model, inputs, file_path):
    
    num_blocks = len(model.vit.blocks)  # Number of blocks to visualize
    fig, ax = plt.subplots(1, 3, figsize=(20, 12))  
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # Initialize variables to store the average attention heatmap
    avg_heatmap = None

    for m, block in enumerate(model.vit.blocks):
        att_mat = block.attn.att_mat
        att_mat_mean_head = torch.mean(att_mat, axis=1)  # Average over the heads
        att_mat_mean_head = att_mat_mean_head.sum(dim=0)
        
        # Process and interpolate the attention heatmap
        att_corner = att_mat_mean_head[0]
        att_corner_reshaped = att_corner.reshape(16, 16, 4)
        
        heatmap_corner = torch.nn.functional.interpolate(
            att_corner_reshaped.unsqueeze(0).unsqueeze(0), 
            scale_factor=(256 // 16, 256 // 16, 64 // 4), 
            mode='trilinear', align_corners=False
        )
        
        # Normalize the heatmap for visualization
        heatmap_norm = (heatmap_corner - heatmap_corner.min()) / (heatmap_corner.max() - heatmap_corner.min())
        heatmap_norm = heatmap_norm.numpy()
        
        # Add the current heatmap to the average
        if avg_heatmap is None:
            avg_heatmap = heatmap_norm
        else:
            avg_heatmap += heatmap_norm
    
    avg_heatmap /= num_blocks
    avg_heatmap = (avg_heatmap - avg_heatmap.min()) / (avg_heatmap.max() - avg_heatmap.min())
        # Prepare the original image for display
    #     inputs_np = inputs[0].permute(1, 2, 3, 0).cpu().numpy()
    #     image_rgb = np.zeros((inputs.shape[2], inputs.shape[3], inputs.shape[4], 3), dtype=np.uint8)
    #     image_rgb[:, :, :, 0] = inputs_np[:, :, :, 0] * 255
    #     image_rgb[:, :, :, 1] = inputs_np[:, :, :, 1] * 255  # Adjust channels as necessary

    #     # Plot original image
    #     ax[0].imshow(np.max(image_rgb, axis=2)) 
    #     ax[0].set_title("MIP Image", fontsize=11)
    #     ax[0].axis('off')  

    #     # Plot average heatmap overlay
    #     ax[1].imshow(np.max(image_rgb, axis=2))
    #     ax[1].imshow(np.max(avg_heatmap[0, 0], axis=2), cmap='jet', alpha=0.5) 
    #     ax[1].set_title("Average Attention Map and MIP Image", fontsize=11)
    #     ax[1].axis('off')

    #     heatmap_color = ax[2].imshow(np.max(avg_heatmap[0, 0], axis=2), cmap='jet', alpha=1) 
    #     ax[2].set_title("Activation Heatmap", fontsize=11)
    #     ax[2].axis('off')

    #     # Add a general title to the figure
    #     fig.suptitle("UNETR Model", fontsize=14)

    #     # Add colorbars only to the last heatmap in each row for clarity
    #     if m == num_blocks - 1:
    #         cbar = fig.colorbar(heatmap_color, ax=ax[2], fraction=0.06, pad=0.04)
    #         cbar.set_label('Attention Intensity', fontsize=10)

    # # Calculate the average heatmap
    # avg_heatmap /= num_blocks

    # # Add a general title to the figure
    # fig.suptitle("UNETR Model Analysis Across All Blocks", fontsize=14)
    
    # plt.tight_layout()
    # fig.savefig(file_path)
    
    return avg_heatmap


def visualization_attn_maps_3D_UNET(model, inputs, outputs, file_path, image_id):   
    
    folder_path = os.path.join(file_path, f'image_{image_id}')
    os.makedirs(folder_path, exist_ok=True)
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
    # heatmap_int = torch.nn.functional.interpolate(heatmap.unsqueeze(0).unsqueeze(0), scale_factor=(256 // 16), mode='trilinear', align_corners=False)
    
    # heatmap_corner_norm = (heatmap_int - heatmap_int.min()) / (heatmap_int.max() - heatmap_int.min())
    # heatmap_corner_norm = heatmap_corner_norm.numpy()
    inputs_np = inputs[0].permute(1, 2, 3, 0).cpu().numpy()
    image_rgb = np.zeros((inputs.shape[2], inputs.shape[3], inputs.shape[4], 3), dtype=np.uint8)
    image_rgb[:, :, :, 0] = inputs_np[:, :, :, 0] * 255
    image_rgb[:, :, :, 1] = inputs_np[:, :, :, 1] * 255 
    
    plt.imshow(np.max(image_rgb, axis=2))
    plt.axis('off')
    plt.savefig(os.path.join(folder_path, f'MIP_{image_id}'), bbox_inches='tight', pad_inches=0)
    plt.close()
    treshold = [0.3,0.15]
    heatmaps = []
    for class_index in range(2):  # Assuming there are two classes
        # Clear gradients
        model.zero_grad()
        
        # Compute gradients for the current class
        gradient = torch.ones_like(outputs[:, class_index])
        outputs[:, class_index].backward(gradient, retain_graph=True)

        # Get gradients and activations
        gradients = model.get_activations_gradient()
        activations = model.get_activations(inputs).detach()

        # Pool the gradients across the channels
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3, 4])

        # Weight the channels by corresponding gradients
        for i in range(gradients.shape[1]):
            activations[:, i, :, :, :] *= pooled_gradients[i]

        # Average the channels of the activations to create the heatmap
        heatmap = torch.mean(activations, dim=1).squeeze()

        # ReLU on the heatmap
        heatmap = F.relu(heatmap)

        # Normalize the heatmap
        heatmap = F.interpolate(heatmap.unsqueeze(0).unsqueeze(0), size=(inputs.shape[2], inputs.shape[3], inputs.shape[4]), mode='trilinear', align_corners=False)
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        heatmap_norm = heatmap_norm.cpu().numpy()

        # Append the normalized heatmap for the current class
        # heatmaps.append(heatmap_norm.cpu().numpy())
        
        heat_mask = (np.max(heatmap_norm[0, 0], axis=2) > treshold[class_index])  
        image_highlighted = np.max(image_rgb, axis=2) * heat_mask[..., np.newaxis]
        
        plt.imshow(image_highlighted)
        plt.axis('off')
        plt.savefig(os.path.join(folder_path, f'Block_{class_index+1}.png'), bbox_inches='tight', pad_inches=0)
        plt.close()
        
    # combined_heatmap = np.maximum.reduce(heatmaps)
    


    # heatmaps = np.array(heatmaps)
    
    # combined_heatmap = np.max(heatmaps, axis=0)

    # Optionally, convert the combined heatmap to a numpy array if necessary for visualization
    
    # fig, ax = plt.subplots(1, 3, figsize=(12, 6))

    # Plot original image
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
    
    # return image_rgb, combined_heatmap
    return image_rgb
    
    
    
    
    # import matplotlib.pyplot as plt
    
    # fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    # plt.subplots_adjust(hspace=0.1, wspace=0.1)
    
    # patches = [(32, 32, 16), (32, 32, 8), (32, 32, 8), (16, 16, 4), (8, 8, 2)]
    
    # avg_heatmap = None
    
    # blocks = [
    #     model.block_1.trans.attention_output.att_mat, 
    #     model.block_2.trans.attention_output.att_mat,
    #     model.block_3.trans.attention_output.att_mat,
    #     model.block_4.trans.attention_output.att_mat,
        
    # ]
    
    # for block, patch_size in zip(blocks, patches):

    #     #attn_map = block.detach().cpu().numpy()

    #     attn_avg = block[0].mean(dim=0)
    #     attn_avg = attn_avg.sum(dim=-1)
    #     print(attn_avg.shape)
    #     att_corner_reshaped = attn_avg.reshape(patch_size)
    #     heatmap = torch.nn.functional.interpolate(att_corner_reshaped.unsqueeze(0).unsqueeze(0), scale_factor=(256 // patch_size[0], 256 // patch_size[1],64 // patch_size[2] ), mode='trilinear')

    #     #heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    #     #heatmap = (heatmap - heatmap.mean()) / (heatmap.std())
        
    #     if avg_heatmap is None:
    #         avg_heatmap = heatmap
    #     else:
    #         avg_heatmap += heatmap
    
    # avg_heatmap /= len(blocks)
    # #avg_heatmap = (avg_heatmap - avg_heatmap.min()) / (avg_heatmap.max() - avg_heatmap.min())

    # avg_heatmap = (avg_heatmap - avg_heatmap.mean()) / (avg_heatmap.std())
    # avg_heatmap = avg_heatmap.numpy()
    
    # inputs_np = inputs[0].permute(1, 2, 3, 0).cpu().numpy()
    # image_rgb = np.zeros((inputs.shape[2], inputs.shape[3], inputs.shape[4], 3), dtype=np.uint8)
    # image_rgb[:, :, :, 0] = inputs_np[:, :, :, 0] * 255
    # image_rgb[:, :, :, 1] = inputs_np[:, :, :, 1] * 255


    # ax[0].imshow(np.max(image_rgb,axis=2))
    # ax[0].set_title("MIP Image", fontsize=14, weight='bold')
    # ax[0].axis('off')
    
    # ax[1].imshow(np.max(image_rgb,axis=2))
    # ax[1].imshow(np.max(avg_heatmap[0,0], axis=2), cmap='jet', alpha=0.25)
    # ax[1].set_title("Average Attention Heatmap and MIP Image", fontsize=14, weight='bold')
    # ax[1].axis('off')
    
    # heatmap_color = ax[2].imshow(np.max(avg_heatmap[0,0], axis=2), cmap='jet')
    # ax[2].set_title("Average Attention Heatmap", fontsize=14, weight='bold')
    # ax[2].axis('off')
    
    # cbar = fig.colorbar(heatmap_color, ax=ax[2], fraction=0.046, pad=0.04)
    # cbar.set_label('Attention Intensity', fontsize=14)
    
    # plt.tight_layout()
    # plt.show()

       
# torch.allclose(attn[0].sum(dim=-1), torch.ones_like(attn[0].sum(dim=-1)),rtol=1e-01, atol=1e-01 )
       
       
# import numpy as np
# data = np.array(attn[0].sum(dim=-1))

# # Compute basic statistics
# mean_val = np.mean(data)
# std_dev = np.std(data)
# min_val = np.min(data)
# max_val = np.max(data)

# # Compute percentiles
# percentiles = np.percentile(data, [25, 50, 75])  # 25th, 50th (median), and 75th percentiles

# print("Mean:", mean_val)
# print("Standard Deviation:", std_dev)
# print("Minimum:", min_val)
# print("Maximum:", max_val)
# print("25th Percentile:", percentiles[0])
# print("Median (50th Percentile):", percentiles[1])
# print("75th Percentile:", percentiles[2])