# import os
# import numpy as np
# import nibabel as nib
# import tifffile
# from tifffile import imread
# from plots import  metrics, save_metrics



# files_name = [
#                 #"Results/Normal/Results_3DUnet_96/com_pretreino",
#               #"Results/Normal/Results_UNETR_96", 
#               #"Results/Normal/Results_FCT_Mod_1_96/Results/Results_FCT_Mod_1_96",
#               "Results/Normal/Results_3DUnet_256/Results_3DUnet_256_com_pretreino", 
#               "Results/Normal/Results_UNETR_256",
#               "Results/Normal/Results_FCT_Mod_1_256"
#               ]


# for file_name in files_name:
    
#     dir1 = file_name + "/visualization"
#     dir2 = "Dataset/Masks"
#     image_files = [f for f in os.listdir(dir1) if f.endswith("tif") ]
#     mask_files = [f for f in os.listdir(dir2) if f.endswith("tif") ]

#     image_files = sorted(image_files )
#     mask_files = sorted(mask_files)

#     images = []
#     masks = []
#     n_images = len(image_files) 

#     print("Downloading Dataset ...")
    
#     fold1_pred = []
#     fold1_target = []
#     fold2_pred = []
#     fold2_target = []
#     fold3_pred = []
#     fold3_target = []
#     metrics_test = []
    
#     for idx in range(0, n_images):
        
#         image_path = os.path.join(dir1, image_files[idx])
#         mask_path = os.path.join(dir2, mask_files[idx])

#         image = imread(image_path)
#         mask = imread(mask_path)
        
#         if idx < 4: 
#             fold1_pred.append(image)
#             fold1_target.append(mask)
#         if idx >= 4 and idx < 6:
#             fold2_pred.append(image)
#             fold2_target.append(mask)
#         if idx >= 6:
#             fold3_pred.append(image)
#             fold3_target.append(mask)

#     print("Calculating Metrics ...")
    
        
#     metrics_test.append(metrics( fold1_target,fold1_pred, 0, " "))
#     metrics_test.append(metrics( fold2_target,fold2_pred, 0, " "))
#     metrics_test.append(metrics( fold3_target,fold3_pred, 0, " "))
    
#     print("Saving Metrics ...")
#     save_metrics(metrics_test, "model", file_name, 1, 0, 0)



    
    # data_FCT_96 = {
    #     "Jaccard Index Golgi": [0.5074, 0.3299, 0.3363],
    #     "Jaccard Index Nuclei": [0.6426, 0.5799, 0.4838],
    #     "Dice Coefficient Golgi": [0.6732, 0.4961, 0.5033],
    #     "Dice Coefficient Nuclei": [0.7824, 0.7341, 0.6521],
    #     "Precision Golgi": [0.9103, 0.3384, 0.8907],
    #     "Precision Nuclei": [0.9245, 0.6814, 0.8247],
    #     "Recall Golgi": [0.5341, 0.9294, 0.3508],
    #     "Recall Nuclei": [0.6782, 0.7956, 0.5392]
    # }
    
    
    # data_3D_UNet_96 = {
    #     "Jaccard Index Golgi": [0.5175, 0.4872, 0.5282],
    #     "Jaccard Index Nuclei": [0.6493, 0.6153, 0.5244],
    #     "Dice Coefficient Golgi": [0.6820, 0.6552, 0.6913],
    #     "Dice Coefficient Nuclei": [0.7874, 0.7619, 0.6880],
    #     "Precision Golgi": [0.9386, 0.5356, 0.6489],
    #     "Precision Nuclei": [0.8205, 0.8097, 0.6741],
    #     "Recall Golgi": [0.5356, 0.8435, 0.7396],
    #     "Recall Nuclei": [0.7568, 0.7194, 0.7026]
    # }
    
    # data_UNETR_96 = {
    #     "Jaccard Index Golgi": [0.3292, 0.5214, 0.4947],
    #     "Jaccard Index Nuclei": [0.6718, 0.6128, 0.5002],
    #     "Dice Coefficient Golgi": [0.4953, 0.6854, 0.6620],
    #     "Dice Coefficient Nuclei": [0.8037, 0.7599, 0.6668],
    #     "Precision Golgi": [0.9874, 0.5735, 0.7782],
    #     "Precision Nuclei": [0.7491, 0.8598, 0.8144],
    #     "Recall Golgi": [0.3306, 0.8517, 0.5759],
    #     "Recall Nuclei": [0.8668, 0.6807, 0.5645]
    # }
        
# import numpy as np

# def calculate_std(data):
#     return {metric: np.std(values, ddof=1) for metric, values in data.items()}

# def main():
#     # Initialize data dictionary with metric names and their corresponding values across three folds
#     # data_3D_U_Net_256 = {
#     #     "Jaccard Index Golgi": [0.5497, 0.5005, 0.5038],
#     #     "Jaccard Index Nuclei": [0.6758, 0.6071, 0.5784],
#     #     "Dice Coefficient Golgi": [0.7094, 0.6671, 0.6700],
#     #     "Dice Coefficient Nuclei": [0.8065, 0.7555, 0.7329],
#     #     "Precision Golgi": [0.9095, 0.5434, 0.6282],
#     #     "Precision Nuclei": [0.8133, 0.8515, 0.7965],
#     #     "Recall Golgi": [0.5815, 0.8638, 0.7178],
#     #     "Recall Nuclei": [0.7999, 0.6790, 0.6787]
#     # }
    
#     # data_UNETR_256 = {
#     #     "Jaccard Index Golgi": [0.5670, 0.4267, 0.5263],
#     #     "Jaccard Index Nuclei": [0.6744, 0.6177, 0.5293],
#     #     "Dice Coefficient Golgi": [0.7237, 0.5982, 0.6897],
#     #     "Dice Coefficient Nuclei": [0.8055, 0.7637, 0.6922],
#     #     "Precision Golgi": [0.9173, 0.4469, 0.7765],
#     #     "Precision Nuclei": [0.7935, 0.8422, 0.7772],
#     #     "Recall Golgi": [0.5975, 0.9044, 0.6203],
#     #     "Recall Nuclei": [0.8179, 0.6986, 0.6240]
#     # }
    
#     # data_FCT_256 = {
#     #     "Jaccard Index Golgi": [0.5981, 0.4739, 0.5395],
#     #     "Jaccard Index Nuclei": [0.6992, 0.5953, 0.5811],
#     #     "Dice Coefficient Golgi": [0.7485, 0.6430, 0.7009],
#     #     "Dice Coefficient Nuclei": [0.8230, 0.7463, 0.7350],
#     #     "Precision Golgi": [0.9256, 0.4992, 0.6510],
#     #     "Precision Nuclei": [0.8630, 0.8838, 0.8456],
#     #     "Recall Golgi": [0.6283, 0.9034, 0.7590],
#     #     "Recall Nuclei": [0.7865, 0.6458, 0.6500]
#     # }

#     # Calculate standard deviation for each metric
#     std_devs = calculate_std(data)
    
#     # Print standard deviations
#     print("Standard Deviations:")
#     for metric, std in std_devs.items():
#         print(f"{metric}: {std:.4f}")

# if __name__ == "__main__":
#     main()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Example data from three models
data_3D_U_Net_256 = {
        "Jaccard Index Golgi": [0.5497, 0.5005, 0.5038],
        "Jaccard Index Nuclei": [0.6758, 0.6071, 0.5784],
        "Dice Coefficient Golgi": [0.7094, 0.6671, 0.6700],
        "Dice Coefficient Nuclei": [0.8065, 0.7555, 0.7329],
        "Precision Golgi": [0.9095, 0.5434, 0.6282],
        "Precision Nuclei": [0.8133, 0.8515, 0.7965],
        "Recall Golgi": [0.5815, 0.8638, 0.7178],
        "Recall Nuclei": [0.7999, 0.6790, 0.6787]
    }
    
data_UNETR_256 = {
        "Jaccard Index Golgi": [0.5670, 0.4267, 0.5263],
        "Jaccard Index Nuclei": [0.6744, 0.6177, 0.5293],
        "Dice Coefficient Golgi": [0.7237, 0.5982, 0.6897],
        "Dice Coefficient Nuclei": [0.8055, 0.7637, 0.6922],
        "Precision Golgi": [0.9173, 0.4469, 0.7765],
        "Precision Nuclei": [0.7935, 0.8422, 0.7772],
        "Recall Golgi": [0.5975, 0.9044, 0.6203],
        "Recall Nuclei": [0.8179, 0.6986, 0.6240]
    }
    
data_FCT_256 = {
        "Jaccard Index Golgi": [0.5981, 0.4739, 0.5395],
        "Jaccard Index Nuclei": [0.6992, 0.5953, 0.5811],
        "Dice Coefficient Golgi": [0.7485, 0.6430, 0.7009],
        "Dice Coefficient Nuclei": [0.8230, 0.7463, 0.7350],
        "Precision Golgi": [0.9256, 0.4992, 0.6510],
        "Precision Nuclei": [0.8630, 0.8838, 0.8456],
        "Recall Golgi": [0.6283, 0.9034, 0.7590],
        "Recall Nuclei": [0.7865, 0.6458, 0.6500]
    }



data_FCT_96 = {
        "Jaccard Index Golgi": [0.5074, 0.3299, 0.3363],
        "Jaccard Index Nuclei": [0.6426, 0.5799, 0.4838],
        "Dice Coefficient Golgi": [0.6732, 0.4961, 0.5033],
        "Dice Coefficient Nuclei": [0.7824, 0.7341, 0.6521],
        "Precision Golgi": [0.9103, 0.3384, 0.8907],
        "Precision Nuclei": [0.9245, 0.6814, 0.8247],
        "Recall Golgi": [0.5341, 0.9294, 0.3508],
        "Recall Nuclei": [0.6782, 0.7956, 0.5392]
    }
    
    
data_3D_UNet_96 = {
        "Jaccard Index Golgi": [0.5175, 0.4872, 0.5282],
        "Jaccard Index Nuclei": [0.6493, 0.6153, 0.5244],
        "Dice Coefficient Golgi": [0.6820, 0.6552, 0.6913],
        "Dice Coefficient Nuclei": [0.7874, 0.7619, 0.6880],
        "Precision Golgi": [0.9386, 0.5356, 0.6489],
        "Precision Nuclei": [0.8205, 0.8097, 0.6741],
        "Recall Golgi": [0.5356, 0.8435, 0.7396],
        "Recall Nuclei": [0.7568, 0.7194, 0.7026]
    }
    
data_UNETR_96 = {
        "Jaccard Index Golgi": [0.3292, 0.5214, 0.4947],
        "Jaccard Index Nuclei": [0.6718, 0.6128, 0.5002],
        "Dice Coefficient Golgi": [0.4953, 0.6854, 0.6620],
        "Dice Coefficient Nuclei": [0.8037, 0.7599, 0.6668],
        "Precision Golgi": [0.9874, 0.5735, 0.7782],
        "Precision Nuclei": [0.7491, 0.8598, 0.8144],
        "Recall Golgi": [0.3306, 0.8517, 0.5759],
        "Recall Nuclei": [0.8668, 0.6807, 0.5645]
    }
        
def prepare_data_for_plotting():
    # Prepare data for plotting
    metrics = ["Dice Coefficient Golgi", "Dice Coefficient Nuclei",
               "Precision Golgi", "Precision Nuclei", "Recall Golgi", "Recall Nuclei"]
    models = ["FCT", "3D U-Net", "UNETR"]
    combined_data = []

    for metric in metrics:
        for model in models:
            if model == "FCT":
                data_list = data_FCT_96[metric]
            elif model == "3D U-Net":
                data_list = data_3D_UNet_96[metric]
            else:
                data_list = data_UNETR_96[metric]
            for data_point in data_list:
                combined_data.append((metric, model, data_point))

    return combined_data

def plot_data(combined_data):
    # Convert list to DataFrame
    df = pd.DataFrame(combined_data, columns=['Metric', 'Model', 'Value'])

    # Set up the matplotlib figure
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Metric', y='Value', hue='Model', data=df, palette='Set2')
    plt.title('Experience 1: Boxplot Comparison of Metrics Across Three Folds for 3D U-Net, UNETR, and FCT Models.')
    plt.xticks(rotation=45)
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.legend(title='Model')
    plt.tight_layout()

    # Show plot
    plt.show()

def main():
    combined_data = prepare_data_for_plotting()
    plot_data(combined_data)

if __name__ == "__main__":
    main()
