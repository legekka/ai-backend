import json
import torch
from modules.models import *
from modules.utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_configs():
    # config.json is a dict containing model names, parameters, paths, etc.
    global config
    with open('models/config.json') as f:
        config = json.load(f)

def load_models():
    global T11
    print("Loading T11...", end="", flush=True)
    T11 = EfficientNetV2S(num_classes=config["T11"]["num_classes"])
    T11 = load_checkpoint(T11, config["T11"]["path"], device=device)
    T11.eval()
    print("Done!")

def main():
    load_configs()
    T11 = EfficientNetV2S(num_classes=config["T11"]["num_classes"])
    # print shape of the model
    print(T11)

if __name__ == '__main__':
    main()