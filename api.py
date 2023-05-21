from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
from modules.utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def tagImage(image):
    img = Image.open(image).convert("RGB")
    img = val_transforms(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    # T11 is the tagger model
    output = Tagger(img)
    # create a list of tags and their probabilities
    output = list(zip(Tagger.classes, output.tolist()[0]))
    # filter out tags with probability < 0.5
    output = list(filter(lambda x: x[1] > 0.5, output))
    # sort by A-Z
    output.sort(key=lambda x: x[0])
    return output


def rateImages(images):
    images = [Image.open(image).convert("RGB") for image in images]
    images = [val_transforms(image) for image in images]
    images = torch.stack(images)
    images = images.to(device)
    # rater is the rater model
    output = Rater(images)
    ratings = []
    # add usernames to output
    for i in range(len(output)):
        ratings.append(list(zip(Rater.usernames, output[i].tolist())))
    # sort
    for i in range(len(ratings)):
        ratings[i].sort(key=lambda x: x[1], reverse=True)
    ratings = list(zip(images, ratings))
    return ratings


app = Flask("AI Backend API Server")


@app.route("/tag", methods=["POST"])
def app_tag():
    image = request.files.get("image")
    if image is None:
        return jsonify({"error": "No image provided"}), 400
    tags = tagImage(image)
    return jsonify({"tags": tags}), 200


@app.route("/rate", methods=["POST"])
def app_rate():
    image = request.files.get("image")
    user = request.form.get("user")
    if user is None:
        return jsonify({"error": "No user provided"}), 400
    if user not in Rater.usernames and user != "all":
        return jsonify({"error": "Invalid user"}), 400
    if image is None:
        return jsonify({"error": "No image provided"}), 400
    ratings = rateImages([image])
    ratings = ratings[0][1]
    if user != "all":
        # filter out all ratings except for the user
        ratings = list(filter(lambda x: x[0] == user, ratings))
        ratings = ratings[0][1]
        return jsonify(ratings), 200
    else:
        return jsonify({"ratings": ratings}), 200


@app.route("/ratebulk", methods=["POST"])
def app_rateBulk():
    images = request.files.getlist("images")
    user = request.form.get("user")
    if user is None:
        user = "all"
    if user not in Rater.usernames and user != "all":
        return jsonify({"error": "Invalid user"}), 400
    if images is None:
        return jsonify({"error": "No images provided"}), 400

    if type(images) != list:
        images = [images]

    ratings = rateImages(images)

    ratings = [rating[1] for rating in ratings]
    if user != "all":
        # filter out all ratings except for the user for each image
        for i in range(len(ratings)):
            ratings[i] = list(filter(lambda x: x[0] == user, ratings[i]))
            ratings[i] = ratings[i][0][1]

        return jsonify({"ratings": ratings}), 200
    else:
        return jsonify({"ratings": ratings}), 200


def main():
    global config
    config = load_configs()
    global Tagger, Rater
    Tagger, Rater = load_models(config, device=device)

    CORS(app)
    app.run(host="0.0.0.0", port="2444")


if __name__ == "__main__":
    main()
