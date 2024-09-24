import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib.table import Table
from data_setup import create_dataset, dataset_generalized
from dataloader import create_dataloader
from create_model import create_model
from engine import train, validate
from plots import plotting_loss, plot_images, metrics, save_metrics, metrics_val, plot_attention
from utils import EarlyStopping, reconstruct_image_from_patches, transformations_train
import os
from torchvision import transforms
import time
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    start_time = time.time()
    parser = argparse.ArgumentParser(description='3D Segmentation Training')
    # print(parser.parse_args)
    # args = main(parser.parse_args())

        # Model-related arguments
    parser.add_argument('--model', type=str, help='3DUnet, UNETR, FCT', default='3DUnet')
    parser.add_argument('--pretrained', action="store_true", help='Use a pre-trained model', default=False)
    parser.add_argument('--model_checkpoint', type=str, help='Path to a pre-trained model checkpoint')
    parser.add_argument('--model_checkpoint_test',action="append" ,type=str, help='Path to a pre-trained model checkpoint for inferencing')
    parser.add_argument('--only_forward', action='store_true', help='Only forward the model', default=False)
    parser.add_argument("--attention_weights", action='store_true', help='Save attention weights', default=False)
    parser.add_argument("--generalize", type=bool, help='Check the model has possibility to generalize', default=False)
    parser.add_argument("--parsial_train", type=bool, help='Train some layers', default=False)

    # Data-related arguments
    parser.add_argument('--data_dir1', type=str, help='Directory containing the Images', default='Dataset/Images/')
    parser.add_argument('--data_dir2', type=str, help='Directory containing the Labels', default='Dataset/Masks/')
    parser.add_argument('--num_classes', type=int, help='Number of segmentation classes', default=2)
    parser.add_argument('--input_channels', type=int, help='Number of input channels', default=2)
    parser.add_argument('--output_channels', type=int, help='Number of input channels', default=2)

    parser.add_argument('--patch_size', type=str, help='Size of input images (e.g., (128, 128, 128) for 3D)', default="64,64,64")
    parser.add_argument('--patch_vit', type=str, help='Size of the patches for the ViT input', default="8,8,4")
    parser.add_argument('--overlap', type=int, help='Percentage od overlap between the patches', default=50)
    parser.add_argument('--batch_size', type=int, help='Batch size for training', default=4)
    parser.add_argument('--augmentations', action='store_true', help='Apply data augmentations', default=False)

    # Training-related arguments
    parser.add_argument('--epochs', type=int, help='Number of training epochs', default=40)
    parser.add_argument('--learning_rate', type=float, help='Learning rate for training', default=0.001)
    parser.add_argument('--patience', type=int, help='Epoch to apply early stoppping', default=10)
    # parser.add_argument('--weight_decay', type=float, help='Weight decay for regularization', default=1e-5)
    parser.add_argument('--optimizer', type=str, help='Optimization algorithm (e.g., Adam, SGD)', default='Adam')
    parser.add_argument('--fold', type=int, help='Fold number for cross-validation', default=3)

    # Logging and saving
    parser.add_argument('--log_dir', type=str, help='Directory for logs and checkpoints')
    parser.add_argument('--save_dir', type=str, help='Directory to save the trained model checkpoints')

    
    args = parser.parse_args()
    args.patch_size = tuple(map(int, args.patch_size.split(',')))   
    args.patch_vit = tuple(map(int, args.patch_vit.split(',')))

    output_dir = f'Results/Results_{args.model}_{args.patch_size[0]}'
    os.makedirs(output_dir, exist_ok=True)
    
    ############## DATA SETUP ##############    
    if args.generalize :
       splits, original_image, original_mask, image_info, padding = dataset_generalized(args.data_dir1, args.data_dir2, args.patch_size, args.input_channels, args.overlap)
       test_splits = [[1,2,3,4,5]]
    else:
        splits, original_image, original_mask, image_info, padding = create_dataset(args.data_dir1, args.data_dir2, args.patch_size, args.input_channels, args.overlap) 
        test_splits = [[7, 8], [5,6], [1,2,3,4]]
        
    ############## Main Loop ##############

    device = 'cuda:1' if torch.cuda.is_available() else "cpu"

    metrics_test = []

    i = 0 
    for i , (X, y, X_test, y_test) in enumerate (splits):
        
        # if i < 2:
        #     continue

    ############## Data Splipt #####################
        
        X_train, X_val, Y_train, y_val = train_test_split(X, y, test_size=0.15)
        
        ############## Class Normalization ##############
        
        count_G = np.sum(Y_train[:,:,:,:,0] == 255)
        count_N = np.sum(Y_train[:,:,:,:,1] == 255)
        class_weights = np.array([(count_G+count_N)/count_G, (count_G+count_N)/count_N])
        class_weights_norm = class_weights / class_weights.sum()
        
        if args.only_forward == False and args.attention_weights == False:
            
        ############## MODEL SETUP ##############
        
            early_stopping = EarlyStopping(patience=args.patience, patch_size=args.patch_size[0], model_name=args.model, output_dir=output_dir ,verbose=True, delta=0)
            print("Fold: ", i )

            model = create_model(
                args.model, 
                args.input_channels, 
                args.output_channels, 
                args.pretrained, 
                args.model_checkpoint, 
                args.patch_size,
                False, 
                device,
                args.patch_vit,
                False)
                
            model.to(device)
            
            ############## Loss and Optimizer ##############
        
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
            criterion = nn.BCEWithLogitsLoss()
            
            ############## Dataloader ##############

            train_dataloader = create_dataloader(X_train, Y_train, args.batch_size, model, True, transformations_train)
            val_dataloader = create_dataloader(X_val, y_val, args.batch_size, model, False, None)
            
            epoch_train_losses = []
            epoch_val_losses = []
            
            if args.parsial_train:
                print("Warm up...")
                # Freeze the whole model
                for param in model.parameters():
                        param.requires_grad = False
                        
                # Unfreeze specific layers
                unfreeze_layers = [
                                    'block_2.trans.conv1.0.weight', 'block_2.trans.conv1.0.bias',
                                    'block_3.trans.conv1.0.weight', 'block_3.trans.conv1.0.bias',
                                    'block_4.trans.conv1.0.weight', 'block_4.trans.conv1.0.bias',
                                    'block_5.trans.conv1.0.weight', 'block_5.trans.conv1.0.bias',
                                    'block_6.trans.conv1.0.weight', 'block_6.trans.conv1.0.bias',
                                    'block_7.trans.conv1.0.weight', 'block_7.trans.conv1.0.bias',
                                    'block_8.trans.conv1.0.weight', 'block_8.trans.conv1.0.bias',
                                    'block_9.trans.conv1.0.weight', 'block_9.trans.conv1.0.bias']

                for name, param in model.named_parameters():
                    if name in unfreeze_layers:
                        param.requires_grad = True
                
                for g in range (0,10):
                            
                    train_loss = train(model, train_dataloader, optimizer, criterion, device, class_weights_norm)
                    print(f"Epoch {g}/{10}, Train Loss: {train_loss:.4f}")
                    
                for param in model.parameters():
                    param.requires_grad = True

                    
            for epoch in range(args.epochs):
                
                print("Training... \n")

                train_loss = train(model, train_dataloader, optimizer, criterion, device, class_weights_norm)
                print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}")
                epoch_train_losses.append(train_loss)

                torch.save(model.state_dict(), f"{output_dir}/{args.model}_{args.patch_size[0]}_last.pth")

                val_loss, targets , predictions = validate(model, val_dataloader, criterion, device, class_weights_norm, False, args.model, output_dir)
                print(f"Validation Loss: {val_loss:.4f}")
                epoch_val_losses.append(val_loss)
                
                metrics_val(targets, predictions, i, args.model, epoch)
                
                args.model_checkpoint_test[i] = early_stopping(val_loss, model, i, epoch)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            plotting_loss(epoch_train_losses, epoch_val_losses, i, epoch, args.model, output_dir, args.batch_size, args.patch_size[0])
            
        elif args.only_forward == True:
            
            model_test = create_model(args.model, args.input_channels, args.output_channels, False, args.model_checkpoint_test[i], args.patch_size, True, device, args.patch_vit, args.attention_weights)
            model_test.to(device)
            
            test_dataloader = create_dataloader(X_test, y_test, args.batch_size, model_test, False, None)
            
            optimizer = optim.Adam(model_test.parameters(), lr=args.learning_rate)
            criterion = nn.BCEWithLogitsLoss()
            
            _ , tar, pre = validate(model_test, test_dataloader, criterion, device, class_weights_norm, args.attention_weights, args.model, output_dir)
            
        elif args.attention_weights == True:
            
            # model1 = create_model("3DUnet", args.input_channels, args.output_channels, False, args.model_checkpoint_test[0], args.patch_size, True, device, args.patch_vit, args.attention_weights)
            # model1.to(device)
            
            # test_dataloader = create_dataloader(X_test, y_test, args.batch_size, model1, False, None)
            
            # optimizer = optim.Adam(model1.parameters(), lr=args.learning_rate)
            # criterion = nn.BCEWithLogitsLoss()
            
            # inputs, attention1 = validate(model1, test_dataloader, criterion, device, class_weights_norm, args.attention_weights, "3D U-Net", output_dir)
            # np.save('Attention_Maps/inputs.npy', inputs)
            
            model2 = create_model("UNETR", args.input_channels, args.output_channels, False, args.model_checkpoint_test[1], args.patch_size, True, device, args.patch_vit, args.attention_weights)
            model2.to(device)
            
            test_dataloader = create_dataloader(X_test, y_test, args.batch_size, model2, False, None)

            optimizer = optim.Adam(model2.parameters(), lr=args.learning_rate)
            criterion = nn.BCEWithLogitsLoss()

            _ , attention2 = validate(model2, test_dataloader, criterion, device, class_weights_norm, args.attention_weights, "UNETR", output_dir)

            # model3 = create_model("FCT_Mod_1_256", args.input_channels, args.output_channels, False, args.model_checkpoint_test[2], args.patch_size, True, device, args.patch_vit, args.attention_weights)
            # model3.to(device)

            # test_dataloader = create_dataloader(X_test, y_test, args.batch_size, model3, False, None)

            # optimizer = optim.Adam(model3.parameters(), lr=args.learning_rate)
            # criterion = nn.BCEWithLogitsLoss()

            # _ , attention3 = validate(model3, test_dataloader, criterion, device, class_weights_norm, args.attention_weights, "FCT", output_dir)
            
            # plot_attention(inputs, attention1, attention2, attention3, output_dir)
            continue 
        
        previous = 0
        predictions_reconstuction = []
        target_original = []
        img_original = []

        for j in test_splits[i]:

            j = (j -1)*4

            indicies, padded_shape, n_patches = image_info[j+1], image_info[j+2], int(image_info[j+3])
            pad = padding[int(j/4)]
            prediction_reconstruction = reconstruct_image_from_patches(pre[previous:previous+n_patches], indicies, padded_shape, pad)
            predictions_reconstuction.append(prediction_reconstruction)
            target_original.append(original_mask[int(j/4)][pad[0]:,pad[1]:,pad[2]:])
            img_original.append(original_image[int(j/4)][pad[0]:,pad[1]:,pad[2]:])
            previous = n_patches + previous

        metrics_test.append(metrics(target_original, predictions_reconstuction, i, args.model))
        # plot_images(img_original, target_original, predictions_reconstuction, i, args.epochs, args.model, output_dir, args.batch_size, args.patch_size[0], test_splits[i])
    end_time = time.time()
    total_time = end_time - start_time
    save_metrics(metrics_test, args.model, output_dir, args.batch_size, args.patch_size[0], total_time)



# import argparse
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# from sklearn.model_selection import train_test_split
# from matplotlib.table import Table
# from data_setup import create_dataset, dataset_generalized
# from dataloader import create_dataloader
# from create_model import create_model
# from engine import train, validate
# from plots import plotting_loss, plot_images, metrics, save_metrics, metrics_val
# from utils import EarlyStopping, reconstruct_image_from_patches, transformations_train
# import os
# from torchvision import transforms
# import time
# import matplotlib.pyplot as plt

# def str2bool(v):
#     if isinstance(v, bool):
#        return v
#     if v.lower() in ('yes', 'true', 't', 'y', '1'):
#         return True
#     elif v.lower() in ('no', 'false', 'f', 'n', '0'):
#         return False
#     else:
#         raise argparse.ArgumentTypeError('Boolean value expected.')

# if __name__ == '__main__':
    
#     start_time = time.time()
#     parser = argparse.ArgumentParser(description='3D Segmentation Training')
#     # print(parser.parse_args)
#     # args = main(parser.parse_args())

#         # Model-related arguments
#     parser.add_argument('--model', type=str, help='3DUnet, UNETR, FCT', default='3DUnet')
#     parser.add_argument('--pretrained', action="store_true", help='Use a pre-trained model', default=False)
#     parser.add_argument('--model_checkpoint',action="append", type=str, help='Path to a pre-trained model checkpoint')
#     parser.add_argument('--model_checkpoint_test',action="append" ,type=str, help='Path to a pre-trained model checkpoint for inferencing')
#     parser.add_argument('--only_forward', action='store_true', help='Only forward the model', default=False)
#     parser.add_argument("--attention_weights", action='store_true', help='Save attention weights', default=False)
#     parser.add_argument("--generalize", type=bool, help='Check the model has possibility to generalize', default=False)
#     #parser.add_argument("--parsial_train", action="append", type=bool, help='Train some layers', default=False)
#     parser.add_argument('--parsial_train', action='append', type=str2bool, help='Partially train some layers', default=[])
#     # Data-related arguments
#     parser.add_argument('--data_dir1', type=str, help='Directory containing the Images', default='Dataset/Images/')
#     parser.add_argument('--data_dir2', type=str, help='Directory containing the Labels', default='Dataset/Masks/')
#     parser.add_argument('--num_classes', type=int, help='Number of segmentation classes', default=2)
#     parser.add_argument('--input_channels', type=int, help='Number of input channels', default=2)
#     parser.add_argument('--output_channels', type=int, help='Number of input channels', default=2)

#     parser.add_argument('--patch_size', type=str, help='Size of input images (e.g., (128, 128, 128) for 3D)', default="64,64,64")
#     parser.add_argument('--patch_vit', type=str, help='Size of the patches for the ViT input', default="8,8,4")
#     parser.add_argument('--overlap', type=int, help='Percentage od overlap between the patches', default=50)
#     parser.add_argument('--batch_size', type=int, help='Batch size for training', default=4)
#     parser.add_argument('--augmentations', action='store_true', help='Apply data augmentations', default=False)

#     # Training-related arguments
#     #parser.add_argument('--epochs', action="append", type=int, help='Number of training epochs', default=40)
#     parser.add_argument('--epochs', action="append", type=int, help='Number of training epochs', default=[])

#     parser.add_argument('--learning_rate', type=float, help='Learning rate for training', default=0.001)
#     parser.add_argument('--patience', type=int, help='Epoch to apply early stoppping', default=10)
#     # parser.add_argument('--weight_decay', type=float, help='Weight decay for regularization', default=1e-5)
#     parser.add_argument('--optimizer', type=str, help='Optimization algorithm (e.g., Adam, SGD)', default='Adam')
#     parser.add_argument('--fold', type=int, help='Fold number for cross-validation', default=3)

#     # Logging and saving
#     parser.add_argument('--log_dir', type=str, help='Directory for logs and checkpoints')
#     parser.add_argument('--save_dir', type=str, help='Directory to save the trained model checkpoints')

    
#     args = parser.parse_args()
#     args.patch_size = tuple(map(int, args.patch_size.split(',')))   
#     args.patch_vit = tuple(map(int, args.patch_vit.split(',')))
    
#     print(args.parsial_train, args.model_checkpoint, args.epochs)

#     output_dir = f'Results/Results_{args.model}_{args.patch_size[0]}'
#     os.makedirs(output_dir, exist_ok=True)
    
#     ############## DATA SETUP ##############    
#     if args.generalize :
#        splits, original_image, original_mask, image_info, padding = dataset_generalized(args.data_dir1, args.data_dir2, args.patch_size, args.input_channels, args.overlap)
#        test_splits = [[1,2,3,4,5]]
#     else:
#         splits, original_image, original_mask, image_info, padding = create_dataset(args.data_dir1, args.data_dir2, args.patch_size, args.input_channels, args.overlap) 
#         test_splits = [[7, 8], [5,6], [1,2,3,4]]
#     ############## Main Loop ##############

#     device = 'cuda:1' if torch.cuda.is_available() else "cpu"

#     metrics_test = []
    

#     i = 0 
#     for i , (X, y, X_test, y_test) in enumerate (splits):
        
#         # if i == 0:
#         #     continue
#         # print(i)
#     ############## Data Splipt #####################
        
#         X_train, X_val, Y_train, y_val = train_test_split(X, y, test_size=0.15)
        
#         ############## Class Normalization ##############
        
#         count_G = np.sum(Y_train[:,:,:,:,0] == 255)
#         count_N = np.sum(Y_train[:,:,:,:,1] == 255)
#         class_weights = np.array([(count_G+count_N)/count_G, (count_G+count_N)/count_N])
#         class_weights_norm = class_weights / class_weights.sum()
        
#         if args.only_forward == False:
            
#         ############## MODEL SETUP ##############
        
#             early_stopping = EarlyStopping(patience=args.patience, patch_size=args.patch_size[0], model_name=args.model, output_dir=output_dir ,verbose=True, delta=0.0003)
#             print("Fold: ", i )

#             model = create_model(
#                 args.model, 
#                 args.input_channels, 
#                 args.output_channels, 
#                 args.pretrained, 
#                 args.model_checkpoint[i], 
#                 args.patch_size,
#                 False, 
#                 device,
#                 args.patch_vit,
#                 False,
#                 i)
                
#             model.to(device)
            
#             ############## Loss and Optimizer ##############
        
#             optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
#             criterion = nn.BCEWithLogitsLoss()
            
#             ############## Dataloader ##############

#             train_dataloader = create_dataloader(X_train, Y_train, args.batch_size, model, True, transformations_train)
#             val_dataloader = create_dataloader(X_val, y_val, args.batch_size, model, False, None)
            
#             epoch_train_losses = []
#             epoch_val_losses = []
#             print(args.parsial_train[i])
#             if bool(args.parsial_train[i]):
                
#                 print("Warm up...")
#                 # Freeze the whole model
#                 for param in model.parameters():
#                         param.requires_grad = False
                        
#                 # Unfreeze specific layers
#                 unfreeze_layers = [
#                                     'block_2.trans.conv1.0.weight', 'block_2.trans.conv1.0.bias',
#                                     'block_3.trans.conv1.0.weight', 'block_3.trans.conv1.0.bias',
#                                     'block_4.trans.conv1.0.weight', 'block_4.trans.conv1.0.bias',
#                                     'block_5.trans.conv1.0.weight', 'block_5.trans.conv1.0.bias',
#                                     'block_6.trans.conv1.0.weight', 'block_6.trans.conv1.0.bias',
#                                     'block_7.trans.conv1.0.weight', 'block_7.trans.conv1.0.bias',
#                                     'block_8.trans.conv1.0.weight', 'block_8.trans.conv1.0.bias',
#                                     'block_9.trans.conv1.0.weight', 'block_9.trans.conv1.0.bias']

#                 for name, param in model.named_parameters():
#                     if name in unfreeze_layers:
#                         param.requires_grad = True
                
#                 for g in range (0,10):
                            
#                     train_loss = train(model, train_dataloader, optimizer, criterion, device, class_weights_norm)
#                     print(f"Epoch {g}/{10}, Train Loss: {train_loss:.4f}")
                    
#                 for param in model.parameters():
#                     param.requires_grad = True
                    
#             for epoch in range(args.epochs[i]):
                
#                 print("Training... \n")

#                 train_loss = train(model, train_dataloader, optimizer, criterion, device, class_weights_norm)
#                 print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}")
#                 epoch_train_losses.append(train_loss)

#                 torch.save(model.state_dict(), f"{output_dir}/{args.model}_{args.patch_size[0]}_last.pth")

#                 val_loss, targets , predictions = validate(model, val_dataloader, criterion, device, class_weights_norm, False)
#                 print(f"Validation Loss: {val_loss:.4f}")
#                 epoch_val_losses.append(val_loss)
                
#                 metrics_val(targets, predictions, i, args.model, epoch)
                
#                 # args.model_checkpoint_test[i] = early_stopping(val_loss, model, i, epoch)
#                 early_stopping(val_loss, model, i, epoch)
#                 if early_stopping.early_stop:
#                     print("Early stopping")
#                     break

#             plotting_loss(epoch_train_losses, epoch_val_losses, i, epoch, args.model, output_dir, args.batch_size, args.patch_size[0])

#         model_test = create_model(args.model, args.input_channels, args.output_channels, False,  args.model_checkpoint_test[i], args.patch_size, True, device, args.patch_vit, args.attention_weights, i)
#         model_test.to(device)
        
#         test_dataloader = create_dataloader(X_test, y_test, args.batch_size, model_test, False, None)
        
#         optimizer = optim.Adam(model_test.parameters(), lr=args.learning_rate)
#         criterion = nn.BCEWithLogitsLoss()
        
#         _ , tar, pre = validate(model_test, test_dataloader, criterion, device, class_weights_norm, args.attention_weights)
            
                
#         previous = 0
#         predictions_reconstuction = []
#         target_original = []
#         img_original = []

#         for j in test_splits[i]:

#             j = (j -1)*4

#             indicies, padded_shape, n_patches = image_info[j+1], image_info[j+2], int(image_info[j+3])
#             pad = padding[int(j/4)]
#             prediction_reconstruction = reconstruct_image_from_patches(pre[previous:previous+n_patches], indicies, padded_shape, pad)
#             predictions_reconstuction.append(prediction_reconstruction)
#             target_original.append(original_mask[int(j/4)][pad[0]:,pad[1]:,pad[2]:])
#             img_original.append(original_image[int(j/4)][pad[0]:,pad[1]:,pad[2]:])
#             previous = n_patches + previous

#         metrics_test.append(metrics(target_original, predictions_reconstuction, i, args.model))
#         #plot_images(img_original, target_original, predictions_reconstuction, i, args.epochs, args.model, output_dir, args.batch_size, args.patch_size[0], test_splits[i])
#     end_time = time.time()
#     total_time = end_time - start_time
#     save_metrics(metrics_test, args.model, output_dir, args.batch_size, args.patch_size[0], total_time)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#     import argparse
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# from sklearn.model_selection import train_test_split
# from matplotlib.table import Table
# from data_setup import create_dataset
# from dataloader import create_dataloader
# from create_model import create_model
# from engine import train, validate
# from plots import plotting_loss, plot_images, metrics, save_metrics, metrics_val
# from utils import EarlyStopping, reconstruct_image_from_patches, transformations_train
# import os
# from torchvision import transforms
# import time
# import matplotlib.pyplot as plt

# if __name__ == '__main__':
    
#     start_time = time.time()
#     parser = argparse.ArgumentParser(description='3D Segmentation Training')
#     # print(parser.parse_args)
#     # args = main(parser.parse_args())
    

#         # Model-related arguments
#     parser.add_argument('--model', type=str, help='3DUnet, UNETR, FCT', default='3DUnet')
#     parser.add_argument('--pretrained', action='store_true', help='Use a pre-trained model', default=False)
#     parser.add_argument('--model_checkpoint', type=str, help='Path to a pre-trained model checkpoint')
#     parser.add_argument('--model_checkpoint_test',action="append" ,type=str, help='Path to a pre-trained model checkpoint for inferencing')
#     parser.add_argument('--only_forward',type=bool, help='Only forward the model', default=False)
#     parser.add_argument("--attention_weights", action='store_true', help='Save attention weights', default=False)

#     # Data-related arguments
#     parser.add_argument('--data_dir1', type=str, help='Directory containing the Images', default='Dataset/Images/')
#     parser.add_argument('--data_dir2', type=str, help='Directory containing the Labels', default='Dataset/Masks/')
#     parser.add_argument('--num_classes', type=int, help='Number of segmentation classes', default=2)
#     parser.add_argument('--input_channels', type=int, help='Number of input channels', default=2)
#     parser.add_argument('--output_channels', type=int, help='Number of input channels', default=2)

#     parser.add_argument('--patch_size', type=str, help='Size of input images (e.g., (128, 128, 128) for 3D)', default="64,64,64")
#     parser.add_argument('--patch_vit', type=str, help='Size of the patches for the ViT input', default="8,8,4")
#     parser.add_argument('--overlap', type=int, help='Percentage od overlap between the patches', default=50)
#     parser.add_argument('--batch_size', type=int, help='Batch size for training', default=4)
#     parser.add_argument('--augmentations', action='store_true', help='Apply data augmentations', default=False)

#     # Training-related arguments
#     parser.add_argument('--epochs', type=int, help='Number of training epochs', default=40)
#     parser.add_argument('--learning_rate', type=float, help='Learning rate for training', default=0.001)
#     parser.add_argument('--patience', type=int, help='Epoch to apply early stoppping', default=10)
#     # parser.add_argument('--weight_decay', type=float, help='Weight decay for regularization', default=1e-5)
#     parser.add_argument('--optimizer', type=str, help='Optimization algorithm (e.g., Adam, SGD)', default='Adam')
#     parser.add_argument('--fold', type=int, help='Fold number for cross-validation', default=3)

#     # Logging and saving
#     parser.add_argument('--log_dir', type=str, help='Directory for logs and checkpoints')
#     parser.add_argument('--save_dir', type=str, help='Directory to save the trained model checkpoints')

    
#     args = parser.parse_args()
#     args.patch_size = tuple(map(int, args.patch_size.split(',')))   
#     args.patch_vit = tuple(map(int, args.patch_vit.split(',')))

#     output_dir = f'Results/Results_{args.model}_{args.patch_size[0]}'
#     os.makedirs(output_dir, exist_ok=True)
    
#     ############## DATA SETUP ##############    

#     splits, original_image, original_mask, image_info, padding = create_dataset(args.data_dir1, args.data_dir2, args.patch_size, args.input_channels, args.overlap) 

#     ############## Main Loop ##############

#     device = 'cuda:1' if torch.cuda.is_available() else "cpu"

#     metrics_test = []
#     test_splits = [[7, 8], [5,6], [1,2,3,4]]

#     i = 0 
#     for i , (X, y, X_test, y_test) in enumerate (splits):
        
       
#     ############## Data Splipt #####################
        
#         X_train, X_val, Y_train, y_val = train_test_split(X, y, test_size=0.15)
        
#         ############## Class Normalization ##############
        
#         count_G = np.sum(Y_train[:,:,:,:,0] == 255)
#         count_N = np.sum(Y_train[:,:,:,:,1] == 255)
#         class_weights = np.array([(count_G+count_N)/count_G, (count_G+count_N)/count_N])
#         class_weights_norm = class_weights / class_weights.sum()
#         print(args.only_forward)
#         if args.only_forward == False:
            
#         ############## MODEL SETUP ##############
        
#             early_stopping = EarlyStopping(patience=args.patience, patch_size=args.patch_size[0], model_name=args.model, output_dir=output_dir ,verbose=True, delta=0.0003)
#             print("Fold: ", i )

#             model = create_model(
#                 args.model, 
#                 args.input_channels, 
#                 args.output_channels, 
#                 args.pretrained, 
#                 args.model_checkpoint, 
#                 args.patch_size,
#                 False, 
#                 device,
#                 args.patch_vit,
#                 False)
#             #torch.cuda.empty_cache()
#             model.to(device)
            
#             ############## Loss and Optimizer ##############
        
#             optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
#             criterion = nn.BCEWithLogitsLoss()
            
#             ############## Dataloader ##############

#             train_dataloader = create_dataloader(X_train, Y_train, args.batch_size, model, True, transformations_train)
#             val_dataloader = create_dataloader(X_val, y_val, args.batch_size, model, False, None)
            
#             epoch_train_losses = []
#             epoch_val_losses = []

#             for epoch in range(args.epochs):
                
#                 print("Training... \n")

#                 train_loss = train(model, train_dataloader, optimizer, criterion, device, class_weights_norm)
#                 print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}")
#                 epoch_train_losses.append(train_loss)

#                 torch.save(model.state_dict(), f"{output_dir}/{args.model}_{args.patch_size[0]}_last.pth")

#                 val_loss, targets , predictions = validate(model, val_dataloader, criterion, device, class_weights_norm, False)
#                 print(f"Validation Loss: {val_loss:.4f}")
#                 epoch_val_losses.append(val_loss)
                
#                 metrics_val(targets, predictions, i, args.model, epoch)
                
#                 #args.model_checkpoint_test[i] = early_stopping(val_loss, model, i, epoch)
#                 early_stopping(val_loss, model, i, epoch)
                
#                 if early_stopping.early_stop:
#                     print("Early stopping")
#                     break

#             plotting_loss(epoch_train_losses, epoch_val_losses, i, epoch, args.model, output_dir, args.batch_size, args.patch_size[0])

#         model_test = create_model(args.model, args.input_channels, args.output_channels, False, f'{output_dir}/{args.model}_{args.patch_size[0]}_{i}_best.pth', args.patch_size, True, device, args.patch_vit, args.attention_weights)
#         model_test.to(device)
        
#         test_dataloader = create_dataloader(X_test, y_test, args.batch_size, model_test, False, None)
        
#         optimizer = optim.Adam(model_test.parameters(), lr=args.learning_rate)
#         criterion = nn.BCEWithLogitsLoss()
        
#         _ , tar, pre = validate(model_test, test_dataloader, criterion, device, class_weights_norm, args.attention_weights)
        
#         previous = 0
#         predictions_reconstuction = []
#         target_original = []
#         img_original = []

#         for j in test_splits[i]:

#             j = (j -1)*4

#             indicies, padded_shape, n_patches = image_info[j+1], image_info[j+2], int(image_info[j+3])
#             pad = padding[int(j/4)]
#             prediction_reconstruction = reconstruct_image_from_patches(pre[previous:previous+n_patches], indicies, padded_shape, pad)
#             predictions_reconstuction.append(prediction_reconstruction)
#             target_original.append(original_mask[int(j/4)][pad[0]:,pad[1]:,pad[2]:])
#             img_original.append(original_image[int(j/4)][pad[0]:,pad[1]:,pad[2]:])
#             previous = n_patches + previous

#         metrics_test.append(metrics(target_original, predictions_reconstuction, i, args.model))
#         plot_images(img_original, target_original, predictions_reconstuction, i, args.epochs, args.model, output_dir, args.batch_size, args.patch_size[0], test_splits[i])
#     end_time = time.time()
#     total_time = end_time - start_time
#     save_metrics(metrics_test, args.model, output_dir, args.batch_size, args.patch_size[0], total_time)