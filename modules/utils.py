import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from modules.models import RaterNN, EfficientNetV2S, RaterNNP

from modules.config import Config

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
    if "dataset_hash" in checkpoint_dict:
        model.dataset_hash = checkpoint_dict["dataset_hash"]
    model.to(device)
    return model


def checkpoint_dataset_hash(path):
    checkpoint_dict = torch.load(path, map_location=torch.device("cpu"))
    if "dataset_hash" in checkpoint_dict:
        return checkpoint_dict["dataset_hash"]
    else:
        return None


def load_image(image_path, unsqueeze=True):
    image = Image.open(image_path).convert("RGB")
    # do val transforms
    image = val_transforms(image)
    if unsqueeze:
        image = image.unsqueeze(0)
    return image


def load_personalized_models(device=None):
    import os

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    T11 = EfficientNetV2S(classes=Config.T11["tags"], device=device)
    T11 = load_checkpoint(
        T11, os.path.join("models", Config.T11["checkpoint_path"]), device=device
    )
    T11.eval()

    ratermodels = []
    for username in Config.rater["usernames"]:
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


def load_models(device=None):
    import os

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Loading T11...", end="", flush=True)
    T11 = EfficientNetV2S(classes=Config.T11["tags"], device=device)
    T11 = load_checkpoint(
        T11, os.path.join("models", Config.T11["checkpoint_path"]), device=device
    )
    T11.eval()
    print("Done!")

    T11_rater = EfficientNetV2S(classes=Config.T11["tags"], device=device)
    T11_rater = load_checkpoint(
        T11_rater,
        os.path.join("models", Config.T11["checkpoint_path"]),
        device=device,
    )
    T11_rater.eval()
    print("Loading Rater...", end="", flush=True)
    rater = RaterNN(T11_rater, usernames=Config.rater["usernames"], device=device)
    rater.rater = load_checkpoint(
        rater.rater,
        os.path.join("models", Config.rater["checkpoint_path"]),
        device=device,
    )
    rater.to(device)
    rater.eval()
    print("Done!")

    return T11, rater


def process_image_for_dataset(image):
    if image.mode != "RGB":
        image = Image.open(image).convert("RGB")
    width, height = image.size
    aspect_ratio = width / height

    if aspect_ratio > 1:
        new_width = 512
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = 512
        new_width = int(new_height * aspect_ratio)

    image_512 = Image.new("RGB", (512, 512), (124, 116, 104))
    image_512.paste(
        image.resize((new_width, new_height)),
        (int((512 - new_width) / 2), int((512 - new_height) / 2)),
    )

    return image_512


def process_image_for_2x(image):
    if image.mode != "RGB":
        image = Image.open(image).convert("RGB")
    width, height = image.size
    aspect_ratio = width / height

    if aspect_ratio > 1:
        new_width = 768
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = 768
        new_width = int(new_height * aspect_ratio)

    image_768 = image.resize((new_width, new_height))

    return image_768


def align_rating(rating):
    return round(round(rating * 6) / 6, 2)

def raternn_up_to_date(discord_ids):
    import os
    import modules.db_functions as dbf
    
    raternn_date = os.path.getmtime("models/RaterNN.pth")

    for discord_id in discord_ids:
        model_hash = checkpoint_dataset_hash(os.path.join("models", f"RaterNNP_{discord_id}.pth"))
        dataset_hash = dbf.generate_dataset_hash(discord_id)
        if model_hash != dataset_hash:
            return False
        
        raternnp_date = os.path.getmtime(os.path.join("models", f"RaterNNP_{discord_id}.pth"))
        if raternnp_date > raternn_date:
            return False
    
    return True