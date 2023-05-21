import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import os

# example data in dataset.json:
# dataset.data[0] = {
#    "image": "0001.jpg",
#     "ratings": [0, 0.2, ..., 1]
# }
# we know exactly how many ratings there are because dataset.usernames contains the list of usernames

class RaterDataset(torch.utils.data.Dataset):
    def __init__(self, data, datapath, transform=None):
        self.data = data
        self.datapath = datapath
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.datapath, self.data[idx]["image"])).convert('RGB')
        if self.transform:
            image = self.transform(image)
        ratings = torch.tensor(self.data[idx]["ratings"])
        return image, ratings

