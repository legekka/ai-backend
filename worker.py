import os
import time
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision import models
import torch
from torch import nn
import json
import argparse

# load config from process argument
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config.json")
args = parser.parse_args()

if not os.path.exists(args.config):
    print("Config file not found!")
    exit()

config = json.load(open(args.config))

device = config["device"]

ratermodels = []

# load users.json and classes.json
users = json.load(open("models/users.json"))
classes = json.load(open("models/tags.json"))["labels"]


class Resnext50(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        resnet = models.resnext50_32x4d(pretrained=False)
        resnet.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=resnet.fc.in_features, out_features=n_classes),
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
        tagger_out = tagger_out.reshape(tagger_out.shape[0], tagger_out.shape[1])
        return self.base_model(tagger_out)


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
val_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

torch.device(device)

print("Loading models...")
taggernn = Resnext50(989)
taggernn.load_state_dict(torch.load("models/TaggerNN4S.pth", map_location=device))
taggernn.to(device)
taggernn.eval()
print("TaggerNN4S loaded!")

print("Loading TaggerNN4S(-2) for RaterNN")
tagger = Resnext50(989)
tagger.load_state_dict(torch.load("models/TaggerNN4S.pth", map_location=device))
tagger.to(device)
tagger.eval()
modules = list(list(tagger.children())[0].children())[:-1]
tagger = nn.Sequential(*modules)
print("TaggerNN4S(-2) loaded!")

for user in users:
    print(f"Loading RaterNN for user {user}")
    model = Rater()
    model.load_state_dict(torch.load(f"models/{user}.pth", map_location=device))
    model.to(device)
    model.eval()
    ratermodels.append(model)
    print(f"RaterNN for user {user} loaded!")

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


def rateImage(image, user):
    img = Image.open(image)
    img = val_transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        rating = float(ratermodels[users.index(user)](img).cpu().numpy()[0][0])
    return rating


def rateImageAll(image):
    img = Image.open(image)
    img = val_transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    with torch.no_grad():
        ratings = []
        for model in ratermodels:
            ratings.append(float(model(img).cpu().numpy()[0][0]))
    return ratings


# Worker client part
import requests


def registerWorker():
    print("Registering worker...")
    url = "http://localhost:2444/registerworker"
    data = config
    response = requests.post(url, json=data)
    print(response.text)


# Web server part
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask("worker")
CORS(app)


@app.route("/tag", methods=["POST"])
def tag():
    image = request.files["image"]
    confidences, labels = tagImage(image)
    return jsonify({"confidences": confidences, "labels": labels})


@app.route("/rate", methods=["POST"])
def rate():
    image = request.files["image"]
    user = request.form["user"]
    if user == "all":
        ratings = rateImageAll(image)
        return jsonify({"rating": ratings, "users": users})
    else:
        rating = rateImage(image, user)
        return jsonify({"rating": rating})


@app.route("/ratebulk", methods=["POST"])
def ratebulk():
    images = request.files.getlist("images")
    user = request.form["user"]
    print(f"RateBulk {len(images)} images for {user}")
    ratings = []
    if user == "all":
        for image in images:
            ratings.append(rateImageAll(image))
        return jsonify({"ratings": ratings, "users": users})
    else:
        for image in images:
            ratings.append(rateImage(image, user))
        return jsonify({"ratings": ratings})


if __name__ == "__main__":
    registerWorker()
    from waitress import serve

    serve(app, host="0.0.0.0", port=config["port"])
