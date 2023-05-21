# test for raterdataset

import json
import torch
from modules.utils import *
from modules.raterdataset import *



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


config = load_configs()
TaggerNN, RaterNN = load_models(config=config, device=device)

print("Model loaded successfully!")

# load the dataset
with open("rater/train.json") as f:
    train_data = json.load(f)

Dataset = RaterDataset(dataset_json=train_data, imagefolder="rater/images", transform=get_val_transforms())

# add test data
testrating = {
    "username": "legekka",
    "rating": 0.9,
}
dataentry = Dataset.add_rating(image="test/a.jpg", user_and_rating=testrating, RaterNN=RaterNN)
print("Added test data successfully!")

print(dataentry)

Dataset.save_dataset("test/train.json")