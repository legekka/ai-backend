from torch import nn
from torchvision import models

class EfficientNetV2S(nn.Module):
    def __init__(self, classes):
        super().__init__()        
        efficientnet = models.efficientnet_v2_s()
        efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=1280, out_features=len(classes))
        )
        self.base_model = efficientnet
        self.sigm = nn.Sigmoid()

        self.classes = classes
    
    def forward(self, x):
        x = self.base_model(x)
        x = self.sigm(x)
        return x

# RaterNN is a specialized model, that takes the output of EfficientNetV2S
# without the classifier: so its input is 1280-dimensional
class RaterNN(nn.Module):
    def __init__(self, effnet, usernames):
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

        # freeze the base model
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)  # flatten the output
        x = self.rater(x)
        return x
