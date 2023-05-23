# test for raterdataset

import json
import torch
from torch.utils.data import DataLoader
from modules.utils import *
from modules.raterdataset import *



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


config = load_configs()
#TaggerNN, RaterNN = load_models(config=config, device=device)

ratermodels = load_personalized_models(config=config, device=device)

print("Model loaded successfully!")


Tdata = RTData("rater/dataset.json", get_val_transforms())

if len(Tdata.full_dataset) == 0:
    print("Creating full dataset...")
    Tdata.create_full_dataset(ratermodels=ratermodels)
    print("Done!")
else:
    
    print("Full dataset already exists with", len(Tdata.full_dataset), "images.")
    print("Verifying full dataset...", end="", flush=True)
    valid = Tdata.verify_full_dataset()
    if valid:
        print("Done!")
    else:
        print("Failed!")
        print("Recreating full dataset...")
        Tdata.create_full_dataset(ratermodels=ratermodels)
        print("Done!")

        print("Verifying full dataset...", end="", flush=True)
        valid = Tdata.verify_full_dataset()
        if valid:
            print("Done!")
        else:
            print("Failed!")
            print("Exiting...")
            exit(1)

Tdata.save_dataset("test/dataset.json")