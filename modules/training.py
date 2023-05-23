import json
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import tqdm
from modules.models import *
from modules.utils import *
from modules.raterdataset import *

class PTrainer:
    def __init__(self, username, tdata=None):
        self.username = username
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.config, self.hparams = self.load_configs()
        self.T11 = self.load_models()
        self.Tdata = self.load_dataset() if tdata is None else tdata
        self.rater = self.create_rater()
        self.train_loader = self.create_train_loader()
        self.optimizer = self.create_optimizer()
        self.scheduler = self.create_scheduler()
        self.criterion = self.create_criterion()

    def load_configs(self):
        with open("models/config.json") as f:
            config = json.load(f)

        with open("rater/hparams_P.json") as f:
            hparams = json.load(f)
        return config, hparams

    def load_models(self):
        T11 = EfficientNetV2S(classes=self.config["T11"]["tags"], device=self.device)
        T11 = load_checkpoint(
            T11, os.path.join("models", self.config["T11"]["checkpoint_path"]), device=self.device
        )
        return T11

    def load_dataset(self):
        Tdata = RTData(
            dataset_json=self.hparams["dataset_json"], transform=get_val_transforms()
        )
        return Tdata

    def create_rater(self):
        rater = RaterNNP(
            self.T11,
            username=self.username,
            device=self.device,
        )
        return rater

    def create_train_loader(self):
        train_loader = DataLoader(
            self.Tdata.get_userdataset(self.username),
            batch_size=self.hparams["batch_size"],
            shuffle=True,
            num_workers=self.hparams["num_workers"],
        )
        return train_loader

    def create_optimizer(self):
        optimizer = torch.optim.Adam(self.rater.parameters(), lr=self.hparams["lr"])
        return optimizer

    def create_scheduler(self):
        t_max = len(self.train_loader) * self.hparams["epochs"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=t_max, eta_min=self.hparams["lr_min"]
        )
        return scheduler

    def create_criterion(self):
        criterion = torch.nn.BCELoss()
        return criterion

    def train(self):
        self.rater.train()
        self.rater.to(self.device)

        print("Starting training")

        for epoch in range(self.hparams["epochs"]):
            batch_losses = []
            loop = tqdm.tqdm(self.train_loader)
            for batch_idx, (images, ratings) in enumerate(loop):
                images = images.to(self.device)
                ratings = ratings.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.rater(images)
                loss = self.criterion(outputs, ratings)
                batch_losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                loop.set_description(
                    f"Epoch [{epoch+1}/{self.hparams['epochs']}] Step [{batch_idx+1}/{len(self.train_loader)}]"
                )
                loop.set_postfix(
                    loss=loss.item(),
                    avg_loss=np.mean(batch_losses),
                    lr=self.scheduler.get_last_lr()[0],
                )

        print("Training complete! Saving model...", flush=True, end="")
        checkpoint_dict = {
            "effnet_checkpoint": os.path.join("models", self.config["T11"]["checkpoint_path"]),
            "model": self.rater.rater.state_dict(),
        }
        checkpoint_name = f'{self.hparams["name"]}_{self.username}.pth'
        torch.save(checkpoint_dict, os.path.join("models", checkpoint_name))
        print("Done!")

        return self.rater