# RaterNN part
from flask_cors import CORS
from flask import Flask, jsonify, request
import os
import time
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision import models
import torch
from torch import nn
import json
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import asyncio

cudadevice = "cuda:0"

# load users.json
with open("models/users.json") as f:
    users = json.load(f)
n_classes = len(users)
with open("models/tags.json") as f:
    classes = json.load(f)["labels"]

ratermodels = []


class Resnext101(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        resnet = models.resnext101_32x8d(pretrained=False)
        resnet.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=resnet.fc.in_features,
                      out_features=n_classes),
        )
        self.base_model = resnet
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        return self.sigm(self.base_model(x))


class Resnext50(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        resnet = models.resnext50_32x4d(pretrained=False)
        resnet.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=resnet.fc.in_features,
                      out_features=n_classes),
        )
        self.base_model = resnet
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        return self.sigm(self.base_model(x))


class Rater(nn.Module):
    def __init__(self):
        super().__init__()
        model = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=1),
            nn.Sigmoid(),
        )
        self.base_model = model

    def forward(self, x):
        tagger_out = tagger(x)
        tagger_out = tagger_out.reshape(
            tagger_out.shape[0], tagger_out.shape[1])
        return self.base_model(tagger_out)


class RDataset(Dataset):
    def __init__(self, training_file, img_folder, transforms):
        self.img_folder = img_folder
        self.transforms = transforms
        self.data = json.load(open(training_file))
        self.len = len(self.data)

        self.imgs = []
        self.ratings = []
        for i in range(self.len):
            self.imgs.append(self.data[i]['image'])
            self.ratings.append(self.data[i]['rating'])

    def __getitem__(self, index):
        rating = self.ratings[index]
        img_path = os.path.join(self.img_folder, self.imgs[index])
        img = Image.open(img_path)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, rating

    def __len__(self):
        return self.len


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
val_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

device = torch.device(cudadevice)

# loading model from saved files
print("Loading models...")
taggernn = Resnext50(989)
taggernn.load_state_dict(torch.load(
    "models/TaggerNN4S.pth", map_location=device))
taggernn.to(device)
taggernn.eval()
print("TaggerNN4S loaded.")

print("Loading TaggerNN4S(-2) for Rater")
tagger = Resnext50(989)  # 989 classes
tagger.load_state_dict(torch.load(
    "models/TaggerNN4S.pth", map_location=device))
tagger.eval()
tagger.to(device)
modules = list(list(tagger.children())[0].children())[:-1]
tagger = nn.Sequential(*modules)


# load models
for user in users:
    print("Loading RaterNN for user: " + user)
    model = Rater()
    model.load_state_dict(torch.load(
        "models/" + user + ".pth", map_location=device))
    model.to(device)
    model.eval()
    ratermodels.append(model)
    print("Model for user " + user + " loaded.")

print("All models loaded!")


def tagImage(image):
    img = Image.open(image)
    img = val_transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    with torch.no_grad():
        raw_preds = taggernn(img).cpu().numpy()[0]
        roundpreds = []
        for i in range(len(raw_preds)):
            roundpreds.append(round(raw_preds[i], 4))
        raw_preds = np.array(raw_preds > 0.5, dtype=float)
        labels = np.array(classes)[np.argwhere(raw_preds > 0.5)[:, 0]]
        confidences = np.array(roundpreds)[np.argwhere(raw_preds > 0.5)[:, 0]]
    confidences = confidences.tolist()
    labels = labels.tolist()
    return confidences, labels


def rateImage(image, username):
    img = Image.open(image)
    img = val_transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        prediction = float(ratermodels[users.index(
            username)](img).cpu().numpy()[0][0])
    return prediction


def rateImage_all(image):
    img = Image.open(image)
    img = val_transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    with torch.no_grad():
        predictions = []
        for model in ratermodels:
            prediction = float(model(img).cpu().numpy()[0][0])
            predictions.append(prediction)
    return predictions


# Web server part

# configure port to 8765
app = Flask("rater")
CORS(app)


@app.route("/tag", methods=["POST"])
def tag():
    image = request.files.get("image")
    confidences, labels = tagImage(image)
    return jsonify({"labels": labels, "confidences": confidences})


@app.route("/rate", methods=["POST"])
def rate():
    image = request.files.get("image")
    user = request.form.get("user")
    print(user)
    if user == "all":
        rating = rateImage_all(image)
        print(rating)
        return jsonify({"rating": rating, "users": users})
    else:
        if user not in users:
            print("User " + user + " not found!")
            return jsonify({"error": "User not found"})
        rating = rateImage(image, user)
        print("Rating for user " + user + ": " + str(rating))
    return jsonify({"rating": rating})


@app.route("/ratebulk", methods=["POST"])
def ratebulk():
    images = request.files.getlist("images")
    user = request.form.get("user")
    ratings = []
    if user == "all":
        for image in images:
            rating = rateImage_all(image)
            ratings.append(rating)
        return jsonify({"ratings": ratings, "users": users})
    else:
        if user not in users:
            print("User " + user + " not found!")
            return jsonify({"error": "User not found"})
        for image in images:
            rating = rateImage(image, user)
            ratings.append(rating)
    return jsonify({"ratings": ratings})


def checkfiles(data):
    # iterate through data and check if all files exist (data.image is the filename)
    for item in data:
        if not os.path.isfile("images/" + item["image"]):
            print("File " + item["image"] + " not found!")
            return False
    return True


def createTrainDataFiles(data):
    for user in users:
        with open("training/" + user + ".json", "w") as outfile:
            traindata = []
            for item in data:
                if item["username"] == user:
                    traindata.append(
                        {"image": item["image"], "rating": item["rating"]})
            json.dump(traindata, outfile)


async def trainAllUser():
    for i, user in enumerate(users):
        print("Training RaterNN for user " + user + " " + str(i + 1) + "/" + str(len(users)))
        await trainUser(user)
    return


async def trainUser(username):
    print("AAA")
    current_training_status["user"] = username
    current_training_status["progress"] = 0
    current_training_status["is_training"] = True
    epoch = 1
    print("Training user " + username + "...")
    training_data = RDataset("training/" + username +
                             ".json", "images", train_transform)
    print("Training data loaded.")
    train_loader = DataLoader(training_data, batch_size=batch_size,
                              num_workers=num_workers, drop_last=True, shuffle=True)
    model = Rater()
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    while True:
        batch_losses = []
        loop = tqdm(train_loader)
        for imgs, targets in loop:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()

            model_result = model(imgs).squeeze(1)
            loss = criterion(model_result, targets.type(torch.float))
            batch_loss_value = loss.item()
            loss.backward()
            optimizer.step()

            batch_losses.append(batch_loss_value)
            loop.set_description(
                f"Epoch [{epoch}/{max_epoch_number}] | Loss: {np.mean(batch_losses):.3f}")
        current_training_status["progress"] = epoch / max_epoch_number
        epoch += 1
        if max_epoch_number < epoch:
            break
    torch.save(model.state_dict(), "models/" + username + ".pth")
    print("User " + username + " trained, model saved!")
    index = users.index(username)
    ratermodels[index].load_state_dict(model.state_dict())
    print("Model updated!")
    current_training_status["is_training"] = False
    # clear up gpu memory
    del model
    torch.cuda.empty_cache()
    return


@app.route("/training/status", methods=["GET"])
def trainingStatus():
    return jsonify(current_training_status)


@app.route("/training/train", methods=["POST"])
def train():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    if current_training_status["is_training"]:
        return jsonify({"error": "Training already in progress"})
    user = request.form.get("user")
    if (user == "all"):
        print("Training all users")
        loop.run_until_complete(trainAllUser())
        return jsonify({"status": "Training all users complete"})

    if user not in users:
        print("User " + user + " not found!")
        return jsonify({"error": "User not found"})
    loop.run_until_complete(trainUser(user))
    return jsonify({"success": "Training complete"})


@app.route("/training/update", methods=["POST"])
def updateTrainingData():
    data = json.loads(request.form.get("data"))
    if not checkfiles(data):
        return jsonify({"error": "Images are missing!"})
    createTrainDataFiles(data)
    print("Training data updated!")
    return jsonify({"success": "Data updated!"})


@app.route("/training/addimages", methods=["POST"])
def addImages():
    images = request.files.getlist("images")
    for image in images:
        image.save("images/" + image.filename)
    return jsonify({"status": "ok"})


@app.route("/training/addimage", methods=["POST"])
def addImage():
    image = request.files.get("image")
    image.save("images/" + image.filename)
    return jsonify({"success": "Image saved"})


@app.route("/updatemodel", methods=["POST"])
def updatemodel():
    modelfile = request.files.get("pth")
    if not modelfile.filename.endswith(".pth"):
        return jsonify({"error": "Invalid file type"})

    user = request.form.get("user")
    if user not in users:
        print("User " + user + " not found!")
        return jsonify({"error": "User not found"})
    # get index of user in users
    index = users.index(user)
    ratermodels[index].load_state_dict(
        torch.load(modelfile, map_location=device))
    # save modelfile into the models folder
    torch.save(ratermodels[index].state_dict(), "models/" + user + ".pth")
    print("Model for user " + user + " updated.")
    return jsonify({"success": "Model updated"})


# Retrainer part

# Fix all seeds to make experiments reproducible
torch.manual_seed(2021)
torch.cuda.manual_seed(2021)
np.random.seed(2021)
random.seed(2021)
torch.backends.cudnn.deterministic = True

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(),
    transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.5, 1.5),
                            shear=None, resample=False,
                            fillcolor=tuple(np.array(np.array(mean)*255).astype(int).tolist())),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

current_training_status = {
    "is_training": False,
    "user": "",
    "progress": 0
}

# Initialize the training parameters.
num_workers = 4  # Number of CPU processes for data preprocessing
lr = 1e-4 * 3  # Learning rate
batch_size = 32  # Batch size
max_epoch_number = 20  # Max epoch number

device = torch.device(cudadevice)


app.run(host="0.0.0.0", port="2444")
