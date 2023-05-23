import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from modules.models import RaterNN, EfficientNetV2S, RaterNNP


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose(
    [
        transforms.Resize((384, 384)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.RandomAffine(
            degrees=15,
            translate=(0.1, 0.1),
            scale=(0.75, 1.25),
            shear=None,
            fill=tuple(np.array(np.array(mean) * 255).astype(int).tolist()),
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)
val_transforms = transforms.Compose(
    [
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)


def get_train_transforms():
    return train_transforms


def get_val_transforms():
    return val_transforms


def load_checkpoint(model, path, device=torch.device("cpu")):
    checkpoint_dict = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint_dict["model"])
    model.to(device)
    return model


def load_image(image_path, unsqueeze=True):
    image = Image.open(image_path).convert("RGB")
    # do val transforms
    image = val_transforms(image)
    if unsqueeze:
        image = image.unsqueeze(0)
    return image



def load_configs():
    import json

    with open("models/config.json") as f:
        config = json.load(f)
    return config


def load_personalized_models(config, device):
    import os

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    T11 = EfficientNetV2S(classes=config["T11"]["tags"], device=device)
    T11 = load_checkpoint(
        T11, os.path.join("models", config["T11"]["checkpoint_path"]), device=device
    )
    T11.eval()

    ratermodels = []
    for username in config["rater"]["usernames"]:
        print(f"Loading RaterNNP for {username}...", end="", flush=True)
        raterp = RaterNNP(
            T11,
            username=username,
            device=device,
        )
        raterp.rater = load_checkpoint(
            raterp.rater,
            os.path.join("models", f"RaterNNP_{username}.pth"),
            device=device,
        )
        raterp.to(device)
        raterp.eval()
        ratermodels.append(raterp)
        print("Done!")
    return ratermodels


def load_models(config, device):
    import os

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Loading T11...", end="", flush=True)
    T11 = EfficientNetV2S(classes=config["T11"]["tags"], device=device)
    T11 = load_checkpoint(
        T11, os.path.join("models", config["T11"]["checkpoint_path"]), device=device
    )
    T11.eval()
    print("Done!")

    T11_rater = EfficientNetV2S(classes=config["T11"]["tags"], device=device)
    T11_rater = load_checkpoint(
        T11_rater,
        os.path.join("models", config["T11"]["checkpoint_path"]),
        device=device,
    )
    T11_rater.eval()
    print("Loading Rater...", end="", flush=True)
    rater = RaterNN(T11_rater, usernames=config["rater"]["usernames"], device=device)
    rater.rater = load_checkpoint(
        rater.rater,
        os.path.join("models", config["rater"]["checkpoint_path"]),
        device=device,
    )
    rater.to(device)
    rater.eval()
    print("Done!")

    return T11, rater
