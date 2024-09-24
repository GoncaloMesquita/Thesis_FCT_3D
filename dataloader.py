import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    
    def __init__(self, images, masks, model, transform):
        self.images = torch.from_numpy(images).permute(0, 4, 1, 2, 3)
        self.masks = torch.from_numpy(masks).permute(0, 4, 1, 2, 3)
        self.model = model
        self.max_image = None
        self.min_image = None
        self.max_mask = None
        self.min_mask = None
        self.transform = transform
 

    def __len__(self):
        
        return len(self.images)

    def __getitem__(self, index):
        
        if self.model.train():
            self.max_image = torch.max(self.images)
            self.min_image = torch.min(self.images)
            self.max_mask = torch.max(self.masks)
            self.min_mask = torch.min(self.masks)
        
        image = (self.images[index] - self.min_image)/(self.max_image - self.min_image)
        mask = (self.masks[index] - self.min_mask)/(self.max_mask - self.min_mask)
        
        if self.transform:
                sample = { "image": image, "label" : mask}
                batch = self.transform(sample)  

                image = batch["image"]
                mask = batch["label"]
            # image = self.transform(image)
        
        return image, mask
    
# Function to create a custom dataset and return dataloader
def create_dataloader(images, masks, batch_size, model, shu, transform):

    dataset = CustomDataset(images, masks, model, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shu)
    
    return  dataloader