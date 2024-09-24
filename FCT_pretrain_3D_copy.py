
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
    ToTensord
)
from Models.FCT_3D_first_1 import FCT
from monai.data import (
    DataLoader,
)
import torch
import torch.nn as nn
import torch.optim as optim
from plots import plotting_loss, plot_images, save_metrics
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import matplotlib.pyplot as plt
import os
from sklearn.metrics import  jaccard_score,  f1_score, precision_score, recall_score

import numpy as np

output_dir = 'Results_FCT_pretrained'
os.makedirs(output_dir, exist_ok=True)

def save_metrics(data, model, output_dir, batch_size, patch_size):

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
        ['Nuclei', f'{avg_jac2:.4f}', f'{avg_dice2:.4f}', f'{avg_pre2:.4f}', f'{avg_recc2:.4f}']
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

    plot_filename = f'Metrics_Test.jpg'
    plot_filepath = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_filepath)

    return


def metrics_pre(targets, predictions, epochs, model, batch_size, type):
    
    num_classes = targets.shape[1]
    
    jac_scores = []
    dice_scores = []
    precision_scores = []
    recall_scores = []

    #with open(f"{output_dir}/metrics.txt", "w") as file:
        #file.write(f"Epoch {epochs}:")
    if epochs %10 == 0:
        
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
        #file.write(output_text + "\n" + "\n")

        #file.close()
    return


class EarlyStopping:
    
    def __init__(self, patience, verbose=True, delta=0):
        
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss, model, dados):
        
        self.dados = dados 
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
            #metrics_save(self.dados)
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


class CustomDataset(Dataset):

    def __init__(self, Images, Labels, transform=None):

        self.images = Images
        self.labels = Labels

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        image = self.images[idx]

        label = create_mask(self.labels[idx])

        return image, label

    
n_samples = 40
patch_size = (96,96,96)

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
    ]
)

learning_rate = 0.00001
batch_size = 8
epochs = 250

device = 'cuda' if torch.cuda.is_available() else "cpu"


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

model = FCT(img_size=patch_size[0])

model.to(device)

early_stopping = EarlyStopping(patience=250, verbose=True, delta=0)

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
        optimizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs, label)
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
    dados = metrics_pre(tar, pre, epoch, "FCT", batch_size, "test")

    early_stopping(val_loss, model, dados)
        
    if early_stopping.early_stop:
        print("Early stopping")
        break
        
    pre = np.vstack(all_predictions)
    tar = np.vstack(all_targets)

    

    plt.plot(range(1, epoch+2), epoch_train_losses, label='Training Loss')
    plt.plot(range(1, epoch+2), epoch_val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss of pre trained FCT Model')
    plt.legend()
    plt.savefig(f"{output_dir}/Train_Val_Pretrained_FCT_{patch_size[0]}")
    plt.close()  








