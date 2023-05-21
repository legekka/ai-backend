import torch
from torch import nn
from torchvision import models
from PIL import Image

class EfficientNetV2S(nn.Module):
    def __init__(self, classes, device):
        super().__init__()        
        efficientnet = models.efficientnet_v2_s()
        efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=1280, out_features=len(classes))
        )
        self.base_model = efficientnet
        self.sigm = nn.Sigmoid()

        self.classes = classes
        self.device = device
    
    def forward(self, x):
        x = self.base_model(x)
        x = self.sigm(x)
        return x
    
    def tagImage(self, image):
        from modules.utils import get_val_transforms
        img = Image.open(image).convert("RGB")
        img = get_val_transforms()(img)
        img = img.unsqueeze(0)
        img = img.to(self.device)
        output = self(img)
        output = list(zip(self.classes, output.tolist()[0]))
        output = list(filter(lambda x: x[1] > 0.5, output))
        output.sort(key=lambda x: x[0])
        return output

# RaterNN is a specialized model, that takes the output of EfficientNetV2S
# without the classifier: so its input is 1280-dimensional
class RaterNN(nn.Module):
    def __init__(self, effnet, usernames, device):
        super().__init__()
        self.rater = nn.Sequential(
            nn.Linear(in_features=1280, out_features=512, bias=True),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.15),
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.05),
            nn.Linear(in_features=256, out_features=128, bias=True),
            nn.BatchNorm1d(128),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.01),
            nn.Linear(in_features=128, out_features=len(usernames), bias=True),
            nn.Sigmoid()
        )
        effnet.base_model.classifier = nn.Identity()
        self.base_model = effnet.base_model
        self.usernames = usernames
        self.device = device

        # freeze the base model
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)  # flatten the output
        x = self.rater(x)
        return x

    def rateImageBatch(self, images):
        from modules.utils import get_val_transforms
        images = [Image.open(image).convert("RGB") for image in images]
        images = [get_val_transforms()(image) for image in images]
        images = torch.stack(images)
        images = images.to(self.device)
        output = self(images)
        # remove tensors
        output = output.tolist()
        ratings = []
        for i in range(len(output)):
            ratings.append(list(zip(self.usernames, output[i])))
        ratings = list(zip(images, ratings))
        ratings = [rating[1] for rating in ratings]
        return ratings
    
    def rateImage(self, image):
        rating = self.rateImageBatch([image])[0]
        return rating