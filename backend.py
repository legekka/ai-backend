# RaterNN part
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

# load users.json
with open('models/users.json') as f:
    users = json.load(f)
n_classes = len(users)
with open('models/tags.json') as f:
    classes = json.load(f)['labels']

ratermodels = []

class Resnext101(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        resnet = models.resnext101_32x8d(pretrained=False)
        resnet.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=resnet.fc.in_features, out_features=n_classes)
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
            nn.Linear(in_features=resnet.fc.in_features, out_features=n_classes)
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
            nn.Sigmoid()
        )
        self.base_model = model

    def forward(self, x):
        tagger_out = tagger(x)
        tagger_out = tagger_out.reshape(tagger_out.shape[0],tagger_out.shape[1])
        return self.base_model(tagger_out)


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

device = torch.device('cuda:1')
# use cuda device 1

# loading model from saved files
print('Loading models...')
taggernn = Resnext50(989)
taggernn.load_state_dict(torch.load('models/TaggerNN4S.pth'))
taggernn.to(device)
taggernn.eval()
print('TaggerNN4S loaded.')

print('Loading TaggerNN4S(-2) for Rater')
tagger = Resnext50(989) # 989 classes
tagger.load_state_dict(torch.load('models/TaggerNN4S.pth'))
tagger.eval()
tagger.to(device)
modules = list(list(tagger.children())[0].children())[:-1]
tagger = nn.Sequential(*modules)


#load models
for user in users:
    print('Loading RaterNN for user: ' + user)
    model = Rater()
    model.load_state_dict(torch.load('models/' + user + '.pth'))
    model.to(device)
    model.eval()
    ratermodels.append(model)
    print('Model for user ' + user + ' loaded.')

print('All models loaded!')


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
        labels = np.array(classes)[np.argwhere(raw_preds > 0.5)[:,0]]
        confidences = np.array(roundpreds)[np.argwhere(raw_preds > 0.5)[:,0]]
    confidences = confidences.tolist()
    labels = labels.tolist()
    return confidences, labels

def rateImage(image, username):
    img = Image.open(image)
    img = val_transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    
    with torch.no_grad():
        prediction = float(ratermodels[users.index(username)](img).cpu().numpy()[0][0])
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
from flask import Flask,jsonify,request
from flask_cors import CORS

# configure port to 8765
app = Flask('rater')
CORS(app)


@app.route('/tag', methods=['POST'])
def tag():
    image = request.files.get('image')
    confidences, labels = tagImage(image)
    return jsonify({'labels': labels, 'confidences': confidences})

@app.route('/rate', methods=['POST'])
def rate():
    image = request.files.get('image')
    user = request.form.get('user')
    print(user)
    if (user == 'all'):
        rating = rateImage_all(image)
        print(rating)
    else:
        if user not in users:
            print('User ' + user + ' not found!')
            return jsonify({'error': 'User not found'})
        rating = rateImage(image, user)
        print('Rating for user ' + user + ': ' + str(rating))
    return jsonify({'rating': rating})

@app.route('/ratebulk', methods=['POST'])
def ratebulk():
    images = request.files.getlist('images')
    user = request.form.get('user')
    if user not in users:
        print('User ' + user + ' not found!')
        return jsonify({'error': 'User not found'})
    ratings = []
    for image in images:
        rating = rateImage(image, user)
        ratings.append(rating)
    return jsonify({'ratings': ratings})

@app.route('/updatemodel', methods=['POST'])
def updatemodel():
    modelfile = request.files.get('pth')
    if (not modelfile.endswith('.pth')):
        return jsonify({'error': 'Invalid file type'})

    user = request.form.get('user')
    if user not in users:
        print('User ' + user + ' not found!')
        return jsonify({'error': 'User not found'})
    # get index of user in users
    index = users.index(user)
    ratermodels[index].load_state_dict(torch.load(modelfile))
    # save modelfile into the models folder
    torch.save(ratermodels[index].state_dict(), 'models/' + user + '.pth')
    print('Model for user ' + user + ' updated.')
    return jsonify({'success': 'Model updated'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='2444')