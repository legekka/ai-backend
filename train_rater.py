import json
import torch
from modules.models import *
from modules.utils import *
from modules.raterdataset import *
import os
from torch.utils.data import DataLoader
import tqdm
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if device.type == "cpu":
    print(
        "WARNING: CPU is being used for training. This will be slow. Consider using a GPU."
    )


def load_configs():
    # config.json is a dict containing model names, parameters, paths, etc.
    global config
    with open("models/config.json") as f:
        config = json.load(f)

    global hparams
    with open("rater/hparams.json") as f:
        hparams = json.load(f)


def load_models():
    print("Loading models...", flush=True, end="")
    T11 = EfficientNetV2S(classes=config["T11"]["tags"])
    T11 = load_checkpoint(
        T11, os.path.join("models", config["T11"]["checkpoint_path"]), device=device
    )

    global rater
    rater = RaterNN(T11, usernames=config["rater"]["usernames"])
    rater.to(device)
    print("Done!")


def save_ratermodel(save_path="models"):
    checkpoint_dict = {
        "effnet_checkpoint": os.path.join("models", config["T11"]["checkpoint_path"]),
        "model": rater.rater.state_dict(),
    }
    checkpoint_name = hparams["name"] + ".pth"
    torch.save(checkpoint_dict, os.path.join(save_path, checkpoint_name))


def main():
    load_configs()
    load_models()

    # load the dataset
    print("Loading dataset...", flush=True, end="")

    train_dataset = RaterDataset(
        dataset_json="rater/train.json", imagefolder="rater/images", transform=get_val_transforms()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=hparams["batch_size"],
        shuffle=True,
        num_workers=hparams["num_workers"],
    )
    print("Done! Loaded {} images.".format(len(train_dataset)), flush=True)
    # set rater to train mode
    rater.train()

    # set up a learning rate scheduler and the optimizer
    optimizer = torch.optim.Adam(rater.parameters(), lr=hparams["lr"])
    t_max = len(train_loader) * hparams["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_max, eta_min=hparams["lr_min"]
    )
    criterion = torch.nn.BCELoss()

    print("Starting training")

    for epoch in range(hparams["epochs"]):
        batch_losses = []
        loop = tqdm.tqdm(train_loader)
        for batch_idx, (images, ratings) in enumerate(loop):
            images = images.to(device)
            ratings = ratings.to(device)

            optimizer.zero_grad()
            outputs = rater(images)
            loss = criterion(outputs, ratings)
            batch_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()

            loop.set_description(
                f"Epoch [{epoch+1}/{hparams['epochs']}] Step [{batch_idx+1}/{len(train_loader)}]"
            )
            loop.set_postfix(loss=loss.item(), avg_loss=np.mean(batch_losses), lr=scheduler.get_last_lr()[0])

    print("Training complete! Saving model...", flush=True, end="")
    save_ratermodel()
    print("Done!")


if __name__ == "__main__":
    main()
