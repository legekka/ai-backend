# TODO: Major revision needed

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
        from modules.utils import load_models

        self.username = username
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.hparams = self.load_hparams()
        self.T11 = load_models(device=self.device)
        self.Tdata = self.load_dataset() if tdata is None else tdata
        self.rater = self.create_rater()
        self.train_loader = self.create_train_loader()
        self.optimizer = self.create_optimizer()
        self.scheduler = self.create_scheduler()
        self.criterion = self.create_criterion()
        self.progress = 0
        self.thread = None
        self.stop_event = None

    def load_hparams(self):
        with open("rater/hparams_P.json") as f:
            hparams = json.load(f)
        return hparams

    def load_dataset(self):
        # TODO: This needs some rework because of the database
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
                if self.stop_event.is_set():
                    print("Training stopped")
                    return

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

                self.progress = (epoch * len(self.train_loader) + batch_idx) / (
                    self.hparams["epochs"] * len(self.train_loader)
                )

        print("Training complete! Saving model...", flush=True, end="")
        from modules.config import Config

        checkpoint_dict = {
            "effnet_checkpoint": os.path.join("models", Config.T11["checkpoint_path"]),
            "model": self.rater.rater.state_dict(),
            "dataset_hash": self.Tdata.get_userdataset(self.username).dataset_hash,
        }
        checkpoint_name = f'{self.hparams["name"]}_{self.username}.pth'
        torch.save(checkpoint_dict, os.path.join("models", checkpoint_name))
        print("Done!")

        # set self to "None" to indicate that training finished
        self.thread = None

    def start_training(self):
        import threading

        if self.thread is not None:
            raise Exception("A training session is already running")

        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.train, args=())
        self.thread.start()

    def stop_training(self):
        if self.thread is None:
            raise Exception("No training session is currently running")

        self.stop_event.set()
        self.thread.join()
        self.thread = None

    def is_training(self):
        return self.thread is not None

    def get_status(self):
        status = {
            "current_user": self.username,
            "is_training": self.is_training(),
            "progress": f"{self.progress*100:.2f}%",
        }
        return status


class Trainer:
    def __init__(self, tdata=None):
        from modules.utils import load_models

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.hparams = self.load_hparams()
        self.T11 = load_models
        self.Tdata = self.load_dataset() if tdata is None else tdata
        self.rater = self.create_rater()
        self.train_loader = self.create_train_loader()
        self.optimizer = self.create_optimizer()
        self.scheduler = self.create_scheduler()
        self.criterion = self.create_criterion()
        self.progress = 0
        self.thread = None
        self.stop_event = None

    def load_hparams(self):
        with open("rater/hparams.json") as f:
            hparams = json.load(f)
        return hparams

    def load_dataset(self):
        Tdata = RTData(
            dataset_json=self.hparams["dataset_json"], transform=get_val_transforms()
        )
        return Tdata

    def create_rater(self):
        from modules.config import Config
        rater = RaterNN(
            self.T11,
            usernames=Config.rater["usernames"],
            device=self.device,
        )
        return rater

    def create_train_loader(self):
        train_loader = DataLoader(
            self.Tdata.full_dataset,
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

    def train(
        self, raternn: RaterNN = None
    ):  # raternn is the currently loaded model in the API. If it's not None, we have to reload its weights with the new ones after training is done
        self.rater.train()
        self.rater.to(self.device)

        print("Starting training")
        for epoch in range(self.hparams["epochs"]):
            batch_losses = []
            loop = tqdm.tqdm(self.train_loader)
            for batch_idx, (images, ratings) in enumerate(loop):
                if self.stop_event.is_set():
                    print("Training stopped")
                    return

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

                self.progress = (epoch * len(self.train_loader) + batch_idx) / (
                    self.hparams["epochs"] * len(self.train_loader)
                )

        print("Training complete! Saving model...", flush=True, end="")
        from modules.config import Config

        checkpoint_dict = {
            "effnet_checkpoint": os.path.join(
                "models", Config.T11["checkpoint_path"]
            ),
            "model": self.rater.rater.state_dict(),
            "dataset_hash": self.Tdata.dataset_hash,
        }
        checkpoint_name = f'{self.hparams["name"]}.pth'
        torch.save(checkpoint_dict, os.path.join("models", checkpoint_name))
        print("Done!")

        # set self to "None" to indicate that training finished
        self.thread = None

        # reload the model in the API if it was loaded before training
        if raternn is not None:
            raternn.rater = load_checkpoint(
                raternn.rater,
                os.path.join("models", Config.rater["checkpoint_path"]),
                device=self.device,
            )
            print("RaterNN model reloaded in API")

    def start_training(self, raternn: RaterNN = None):
        import threading

        if self.thread is not None:
            raise Exception("A training session is already running")

        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.train, args=(raternn,))
        self.thread.start()

    def stop_training(self):
        if self.thread is None:
            raise Exception("No training session is currently running")

        self.stop_event.set()
        self.thread.join()
        self.thread = None

    def is_training(self):
        return self.thread is not None

    def get_status(self):
        status = {
            "current_user": "all",
            "is_training": self.is_training(),
            "progress": f"{self.progress*100:.2f}%",
        }
        return status
