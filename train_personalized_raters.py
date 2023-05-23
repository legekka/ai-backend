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
    with open("rater/hparams_P.json") as f:
        hparams = json.load(f)


def load_models():
    global T11
    print("Loading models...", flush=True, end="")
    T11 = EfficientNetV2S(classes=config["T11"]["tags"], device=device)
    T11 = load_checkpoint(
        T11, os.path.join("models", config["T11"]["checkpoint_path"]), device=device
    )


def save_ratermodel(rater, username, save_path="models"):
    checkpoint_dict = {
        "effnet_checkpoint": os.path.join("models", config["T11"]["checkpoint_path"]),
        "model": rater.rater.state_dict(),
    }
    checkpoint_name = f'{hparams["name"]}_{username}.pth'
    torch.save(checkpoint_dict, os.path.join(save_path, checkpoint_name))


def main():
    load_configs()
    load_models()

    # load the dataset
    print("Loading dataset...", flush=True, end="")

    Tdata = RTData(
        dataset_json=hparams["dataset_json"], transform=get_val_transforms()
    )
    print("Done!", flush=True)

    # We are training multiple models here, we will simply iterate through the usernames
    for username in Tdata.usernames:
        print("Training RaterNNP for user {}".format(username))

        # create a new rater model
        rater = RaterNNP(
            T11,
            username=username,  
            device=device,  
        )
        print("Model created. Loading dataset...", flush=True, end="")

        train_loader = DataLoader(
            Tdata.get_userdataset(username),
            batch_size=hparams["batch_size"],
            shuffle=True,
            num_workers=hparams["num_workers"],
        )

        print("Done! Loaded {} images.".format(len(Tdata.get_userdataset(username))), flush=True)

        rater.train()
        rater.to(device)

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
                loop.set_postfix(
                    loss=loss.item(),
                    avg_loss=np.mean(batch_losses),
                    lr=scheduler.get_last_lr()[0],
                )

        print("Training complete! Saving model...", flush=True, end="")
        save_ratermodel(rater, username)
        print("Done!")


if __name__ == "__main__":
    main()
