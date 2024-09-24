import matplotlib.pyplot as plt
import numpy as np  
from sklearn.metrics import  jaccard_score,  f1_score, precision_score, recall_score
import os
import nibabel as nib
import tifffile


def metrics_val(target, predictions, fold, model, epoch):
    
    if epoch % 20 == 0:
        
        jac1 = jaccard_score(target[:,0].flatten(), predictions[:,0].flatten(), average='binary')

        dice1 = f1_score(target[:,0].flatten(), predictions[:,0].flatten(), average='binary')

        pre1 = precision_score(target[:,0].flatten(), predictions[:,0].flatten(), average='binary')

        recc1 = recall_score(target[:,0].flatten(), predictions[:,0].flatten(), average='binary')

        jac2 = jaccard_score(target[:,1].flatten(), predictions[:,1].flatten(), average='binary')

        dice2 = f1_score(target[:,1].flatten(), predictions[:,1].flatten(), average='binary')

        pre2 = precision_score(target[:,1].flatten(), predictions[:,1].flatten(), average='binary')

        recc2 = recall_score(target[:,1].flatten(), predictions[:,1].flatten(), average='binary')

        data = [ jac1, dice1, pre1, recc1, jac2, dice2, pre2, recc2]

        print(f'Validation Data Set. \n Jaccar Index Golgi: {jac1:.4f}, Jaccar Index Nuclei: {jac2:.4f} \n Dice Coefficient Golgi: {dice1:.4f}, Dice Coefficient Nuclei: {dice2:.4f} \n Precision Golgi: {pre1:.4f}, Precision Nuclei: {pre2:.4f} \n Recall Golgi: {recc1:.4f}, Recall Nuclei: {recc2:.4f}')
    
    return 


def metrics(targets, predictions, fold, model):

    tar1 = []
    pred1 = []
    tar2 = []
    pred2 = []

    for target, prediction in zip(targets, predictions):
        # Normalize target array
        target = (target - np.min(target)) / (np.max(target) - np.min(target))
        prediction = (prediction - np.min(prediction)) / (np.max(prediction) - np.min(prediction))
        
        # Extract and flatten the data for the first channel
        seq1 = target[:,:,:,0].flatten()
        pred = prediction[:,:,:,0].flatten()
        
        tar1.append(seq1)
        pred1.append(pred)

    # Concatenate and convert data type
    tar1 = np.concatenate(tar1).astype(np.float32)
    pred1 = np.concatenate(pred1).astype(np.float32)

    for target, prediction in zip(targets, predictions):
        # Normalize target array again for the second channel
        target = (target - np.min(target)) / (np.max(target) - np.min(target))
        prediction = (prediction - np.min(prediction)) / (np.max(prediction) - np.min(prediction))
        
        # Extract and flatten the data for the second channel
        seq2 = target[:,:,:,1].flatten()
        pred = prediction[:,:,:,1].flatten()
        
        tar2.append(seq2)
        pred2.append(pred)

    # Concatenate and convert data type
    tar2 = np.concatenate(tar2).astype(np.float32)
    pred2 = np.concatenate(pred2).astype(np.float32)


    # Compute metrics using correct variable names
    dice_golgi = f1_score(tar1, pred1, average='binary')
    recall_golgi = recall_score(tar1, pred1, average='binary')
    precision_golgi = precision_score(tar1, pred1, average='binary')
    jaccard_golgi = jaccard_score(tar1, pred1, average='binary')

    dice_nuclei = f1_score(tar2, pred2, average='binary')
    recall_nuclei = recall_score(tar2, pred2, average='binary')
    precision_nuclei = precision_score(tar2, pred2, average='binary')
    jaccard_nuclei = jaccard_score(tar2, pred2, average='binary')

    
    data = [ jaccard_golgi, dice_golgi , precision_golgi, recall_golgi , jaccard_nuclei, dice_nuclei, precision_nuclei, recall_nuclei]
    
    print(f'Test Data Set. \n Jaccar Index Golgi: {data[0]:.4f}, Jaccar Index Nuclei: {data[4]:.4f} \n Dice Coefficient Golgi: {data[1]:.4f}, Dice Coefficient Nuclei: {data[5]:.4f} \n Precision Golgi: {data[2]:.4f}, Precision Nuclei: {data[6]:.4f} \n Recall Golgi: {data[3]:.4f}, Recall Nuclei: {data[7]:.4f}')
    
    return data


def save_metrics(data, model, output_dir, batch_size, patch_size, time):

    data =  np.array(data)
    avg_jac1 = data[:,0].mean()
    avg_dice1 = data[:,1].mean()
    avg_pre1 = data[:,2].mean()
    avg_recc1 = data[:,3].mean()
    avg_jac2 = data[:,4].mean()
    avg_dice2 = data[:,5].mean()
    avg_pre2 = data[:,6].mean()
    avg_recc2 = data[:,7].mean()
        
    metric = [
        [f'AVG metrics Model: {model}', 'Jaccard Index', 'Dice Coefficient', 'Precision', 'Recall'],
        ['Golgi', f'{avg_jac1:.4f}', f'{avg_dice1:.4f}', f'{avg_pre1:.4f}', f'{avg_recc1:.4f}'],
        ['Nuclei', f'{avg_jac2:.4f}', f'{avg_dice2:.4f}', f'{avg_pre2:.4f}', f'{avg_recc2:.4f}'],
        ["Time", f'{time:.4f}', "", "", ""]
        ]

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))

    # Hide axis
    ax.axis('off')

    # Create a table
    table = ax.table(cellText=metric, loc='center', cellLoc='center', cellColours=None)

    # Add style to the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)

    # Add the table to the axis
    ax.add_table(table)

    # Adjust the cell heights for better formatting
    table.auto_set_column_width([0, 1, 2, 3, 4])
    table.scale(1, 2.5)

    plot_filename = f'Avg_Metrics_Test-PS:{patch_size}_B{batch_size}'
    plot_filepath = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_filepath, format='png')

    return
                       

def plotting_loss(t_loss, v_loss, fold, epochs, model, output_dir, batch_size, patch_size):
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(0, epochs + 1), t_loss, label='Training Loss', marker='o')
    plt.plot(range(0, epochs + 1), v_loss, label='Validation Loss', marker='o')
    plt.title(f'Training and Validation Loss. Fold: {fold}, Epochs: {epochs}, Batch Size: {batch_size}.jpg')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plot_filename = f'Losses-PS:{patch_size}_B{batch_size}_{fold}'
    plot_filepath = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_filepath)
    plt.show()

    
def plot_images(images, targets, predictions, fold, epochs, model, output_dir, batch_size, patch_size, splits):


    for i , (image, target, prediction) in enumerate(zip(images, targets, predictions)):
        
        fig , axis = plt.subplots(1, 3, figsize=(15, 5))
        
        target_rgb = np.zeros((target.shape[0], target.shape[1], target.shape[2], 3), dtype=np.uint8)
        prediction_rgb = np.zeros((prediction.shape[0], prediction.shape[1], prediction.shape[2], 3), dtype=np.uint8)
        image_rgb = np.zeros((image.shape[0], image.shape[1], image.shape[2], 3), dtype=np.uint8)

        target_rgb[:,:,:,0] = target[:,:,:,0]
        target_rgb[:,:,:,1] = target[:,:,:,1]

        prediction_rgb[:,:,:,0] = (prediction[:,:,:,0]*255).astype('uint8')
        prediction_rgb[:,:,:,1] = (prediction[:,:,:,1]*255).astype('uint8')

        image_rgb[:,:,:,0] = image[:,:,:,0]
        image_rgb[:,:,:,1] = image[:,:,:,1]
        
        tifffile.imwrite(f'{output_dir}/prediction_mask_{splits[i]}_{model}.tif', prediction_rgb, photometric='rgb')
        
        axis[0].imshow(np.max(image_rgb, axis=2))
        axis[0].set_title('Image')   
        
        axis[1].imshow(np.max(target_rgb, axis=2))
        axis[1].set_title('Ground Truth')

        axis[2].imshow(np.max(prediction_rgb, axis=2))
        axis[2].set_title('Predictions')

        plt.tight_layout()
        plot_filename = f'Image_vs_GT_vs_Prediction-Image{splits[i]}_PS{patch_size}_B{batch_size}_F{fold}.jpg'
        plot_filepath = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_filepath)
        # plt.show()
    
    return 


def plot_attention(inputs, attn1, attn2, attn3, dir):
    
    num_images = inputs.shape[0]  # Assuming the first dimension is the batch size
    fig, ax = plt.subplots(num_images, 4, figsize=(16, num_images * 4))  # Adjust size for vertical layout
    
    for i in range(num_images):
        input_image = inputs[i].permute(1, 2, 0).cpu().numpy()  # Assuming CHW format and converting to HWC for imshow

        # Normalize and convert attention maps if necessary
        attn_map1 =attn1[i]
        attn_map2 =attn2[i]
        attn_map3 =attn3[i]

        # Plotting input images
        ax[i, 0].imshow(np.max(input_image, axis=2))
        ax[i, 0].axis('off')
        ax[i, 0].set_title('MIP Image' if i == 0 else "")

        # Overlay Attention Map 1
        ax[i, 1].imshow(np.max(input_image, axis=2), cmap='gray')
        ax[i, 1].imshow(np.max(attn_map1, axis=2), cmap='jet', alpha=0.6)  # semi-transparent
        ax[i, 1].axis('off')
        ax[i, 1].set_title('3D U-Net Attention Map' if i == 0 else "")

        # Overlay Attention Map 2
        ax[i, 2].imshow(np.max(input_image, axis=2), cmap='gray')
        ax[i, 2].imshow(np.max(attn_map2, axis=2), cmap='jet', alpha=0.6)
        ax[i, 2].axis('off')
        ax[i, 2].set_title('UNETR Attention map' if i == 0 else "")

        # Overlay Attention Map 3
        ax[i, 3].imshow(np.max(input_image, axis=2), cmap='gray')
        ax[i, 3].imshow(np.max(attn_map3, axis=2), cmap='jet', alpha=0.6)
        ax[i, 3].axis('off')
        ax[i, 3].set_title('FCT Attention map' if i == 0 else "")

    # Colorbar for the heatmaps
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])  # Adjust these values for your layout
    norm = plt.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap="jet", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax)

    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the rect values if necessary
    plt.show()
    fig.savefig(dir)  # Save the figure
    return



