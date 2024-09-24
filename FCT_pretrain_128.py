from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    SpatialPadd
)
from Models.Created.FCT_3D_Study_1 import FCT_Mod_1
from Models.Created.FCT_3D_Study_3 import FCT_Mod_3

from monai.data import (
    DataLoader,
)
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from plots import plotting_loss, plot_images, save_metrics
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import matplotlib.pyplot as plt
import os
from sklearn.metrics import  jaccard_score,  f1_score, precision_score, recall_score

import numpy as np

output_dir = 'Results/Results_FCT_Mod_2_pretrained_96'
os.makedirs(output_dir, exist_ok=True)


def metrics_pre(targets, predictions, epochs, model, batch_size, type):
    
    if epochs % 30 == 0 :
        
        num_classes = targets.shape[1]

        jac_scores = []
        dice_scores = []
        precision_scores = []
        recall_scores = []

        for class_index in range(num_classes):

            jac = jaccard_score(targets[:, class_index].flatten(), predictions[:, class_index].flatten(), average='binary', zero_division=1)
            dice = f1_score(targets[:, class_index].flatten(), predictions[:, class_index].flatten(), average='binary', zero_division=1)
            pre = precision_score(targets[:, class_index].flatten(), predictions[:, class_index].flatten(), average='binary', zero_division=1)
            recc = recall_score(targets[:, class_index].flatten(), predictions[:, class_index].flatten(), average='binary', zero_division=1)

            jac_scores.append(jac)
            dice_scores.append(dice)
            precision_scores.append(pre)
            recall_scores.append(recc)

            output_text = f'Test Data Set. CLASS {class_index}: \n JI: {jac:.4f}, DC: {dice:.4f},  Precision: {pre:.4f}, Recall: {recc:.4f}'
            print(output_text)
    return

def plot_images( targets, predictions, epochs):
    

    fig , axis = plt.subplots(1, 3, figsize=(15, 5))
    prediction = np.argmax(predictions, axis=1)

    for j in range (0, targets.shape[0], 4):
        
        target_slice = targets[ j, 0, :, : , 35]  # Get the target slice
        prediction_slice = prediction[ j, 0, :, : , 35]  # Get the corresponding prediction slice
        # image_slice = images[ j, 0, :, : , 35]  # Get the corresponding image slice

        # axis[0].imshow(image_slice)
        # axis[0].set_title('Image')   
        
        axis[1].imshow(target_slice)
        axis[1].set_title('Ground Truth')

        axis[2].imshow(prediction_slice)
        axis[2].set_title('Predictions')

        plt.tight_layout()
        plot_filename = f'Image_vs_GT_vs_Prediction-S:{i}_{epochs}.jpg'
        plot_filepath = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_filepath)
        plt.show()
    
    return 

class EarlyStopping:
    
    def __init__(self, patience, verbose=True, delta=0):
        
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss, model):

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
        torch.save(model.state_dict(), f"{output_dir}/Pretrained_FCT_{patch_size[0]}_best.pth")

def create_mask(y):

    masks = torch.zeros((14, y.shape[1], y.shape[2], y.shape[3]), dtype=torch.float32, device=device)

    masks[0, :, :] = torch.where(y == 0, 1, 0)
    masks[1, :, :] = torch.where(y == 1, 1, 0)
    masks[2, :, :] = torch.where(y == 2, 1, 0)
    masks[3, :, :] = torch.where(y == 3, 1, 0)
    masks[4, :, :] = torch.where(y == 4, 1, 0)
    masks[5, :, :] = torch.where(y == 5, 1, 0)
    masks[6, :, :] = torch.where(y == 6, 1, 0)
    masks[7, :, :] = torch.where(y == 7, 1, 0)
    masks[8, :, :] = torch.where(y == 8, 1, 0)
    masks[9, :, :] = torch.where(y == 9, 1, 0)
    masks[10, :, :] = torch.where(y == 10, 1, 0)
    masks[11, :, :] = torch.where(y == 11, 1, 0)
    masks[12, :, :] = torch.where(y == 12, 1, 0)
    masks[13, :, :] = torch.where(y == 13, 1, 0)

    return masks


n_samples = 1
patch_size = (96,96,96)
patch_vit = (3,3,3) 


class CustomDataset(Dataset):

    def __init__(self, Images, Labels, transform=None):

        self.images = Images
        self.labels = Labels

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        image = self.images[idx].repeat(2,1,1,1)

        label = create_mask(self.labels[idx])

        return image, label


train_transforms = Compose(
    
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=1,
            neg=1,
            num_samples=n_samples,
            image_key="image",
            image_threshold=0,
        ),
        # SpatialPadd(keys=["image", "label"], spatial_size=(256, 256, 64), method='symmetric'),

        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.10,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
        ToTensord(keys=["image", "label"]),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=1,
            neg=1,
            num_samples=n_samples,
            image_key="image",
            image_threshold=0,
        ),
        # SpatialPadd(keys=["image", "label"], spatial_size=(256, 256, 64), method='symmetric'),

    ]
)

learning_rate = 0.0001
batch_size = 1
epochs = 300

device = 'cuda:1' if torch.cuda.is_available() else "cpu"


npz_file_path_image = []
npz_file_path_label = []

datalist_image = "RawData/Training/img"
npz_files = [f for f in os.listdir(datalist_image) if f.endswith('.gz')]
npz_files_images = sorted(npz_files)

for npz_file in npz_files_images:

    npz_file_path_image.append(os.path.join(datalist_image, npz_file) )

train_image_files = npz_file_path_image[0:25]
val_image_file = npz_file_path_image[25:]

datalist_label = "RawData/Training/label"
npz_files = [f for f in os.listdir(datalist_label) if f.endswith('.gz')]
npz_files_label = sorted(npz_files)

for npz_file in npz_files_label:
    npz_file_path_label.append(os.path.join(datalist_label, npz_file))

train_label_files = npz_file_path_label[0:25]
val_label_file = npz_file_path_label[25:]

train_files = {"image": train_image_files, "label": train_label_files}
val_files = {"image": val_image_file, "label": val_label_file}

train_images = []
train_labels = []

for idx in range(len(train_files["image"])):

    sample = { "image": train_files["image"][idx], "label" : train_files["label"][idx]}
    batch = train_transforms(sample)  
    
    for crop in batch:
            train_images.append(crop["image"])
            train_labels.append(crop["label"])

train_images = torch.stack(train_images)
train_labels = torch.stack(train_labels)

val_images = []
val_labels = []

for idx in range(len(val_files["image"])):

    sample = { "image": val_files["image"][idx], "label" : val_files["label"][idx]}
    batch = val_transforms(sample)  

    for crop in batch:
            val_images.append(crop["image"])
            val_labels.append(crop["label"])

val_images = torch.stack(val_images)
val_labels = torch.stack(val_labels)

_ ,weights = np.unique(train_labels, return_counts = True)
class_weights = weights.sum() / weights 
class_weights_norm = class_weights / class_weights.sum()

train_ds = CustomDataset(train_images, train_labels, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

val_ds = CustomDataset(val_images, val_labels, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

datasets = []
dataloaders = []

epoch_train_losses = []
epoch_val_losses = []

model = FCT_Mod_3(in_channels=2, out_channels=14)

model.to(device)

early_stopping = EarlyStopping(patience=250, verbose=True)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()

epoch_train_losses = []
epoch_val_losses = []

for epoch in range(epochs):

    print(f"Epoch {epoch+1}/{epochs}:")

    running_loss = 0.0
    model.train()
    
    for image, label in train_loader:   
        
        image, label = image.to(device, dtype=torch.float32), label.to(device, dtype=torch.float32)
        print(image.shape)
        optimizer.zero_grad()
        outputs = model(image)
        loss = 0
        for i in range(0,outputs.shape[1]):
            loss += criterion(outputs[:, i], label[:, i]) * class_weights_norm[i]
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    train_loss =  running_loss / len(train_loader)
    epoch_train_losses.append(train_loss)
    
    print(f"Train Loss: {train_loss:.4f}")
    torch.save(model.state_dict(), f"{output_dir}/Pretrained_FCT_{patch_size[0]}_last.pth")
    
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
       
        for inp, out in val_loader:
            
            inputs, targets = inp.to(device, dtype=torch.float32), out.to(device, dtype=torch.float32)
            outputs = model(inputs)
            loss = 0
            for i in range(0,outputs.shape[1]):
                loss += criterion(outputs[:, i], targets[:, i]) * class_weights_norm[i]

            running_loss += loss.item()
            outputs =  torch.sigmoid(outputs)
            predictions = (outputs > 0.5).float().cpu().numpy()
            target = targets.cpu().numpy()

            all_predictions.append(predictions)
            all_targets.append(target)

    val_loss = running_loss / len(val_loader)
    epoch_val_losses.append(val_loss)
    print(f"Validation Loss:, {val_loss:.4f}")

    early_stopping(val_loss, model)
        
    if early_stopping.early_stop:
        print("Early stopping")
        break
        
    pre = np.vstack(all_predictions)
    tar = np.vstack(all_targets)

    metrics_pre(tar, pre, epoch, "FCT", batch_size, "test")
    #plot_images(tar, pre, epoch)

    plt.plot(range(1, epoch+2), epoch_train_losses, label='Training Loss')
    plt.plot(range(1, epoch+2), epoch_val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss of pre trained FCT Model')
    plt.legend()
    plt.savefig(f"{output_dir}/Train_Val_Pretrained_FCT_{patch_size[0]}")
    plt.close()  








