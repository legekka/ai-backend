from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
from modules.utils import load_configs, load_models
from modules.raterdataset import RaterDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

app = Flask("AI Backend API Server")


@app.route("/tag", methods=["POST"])
def app_tag():
    image = request.files.get("image")
    if image is None:
        return jsonify({"error": "No image provided"}), 400
    tags = TaggerNN.tagImage(image)
    return jsonify({"tags": tags}), 200


@app.route("/rate", methods=["POST"])
def app_rate():
    image = request.files.get("image")
    user = request.form.get("user")
    if user is None:
        return jsonify({"error": "No user provided"}), 400
    if user not in RaterNN.usernames and user != "all":
        return jsonify({"error": "Invalid user"}), 400
    if image is None:
        return jsonify({"error": "No image provided"}), 400
    ratings = RaterNN.rateImage(image)
    print(ratings)
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
    if user not in RaterNN.usernames and user != "all":
        return jsonify({"error": "Invalid user"}), 400
    if images is None:
        return jsonify({"error": "No images provided"}), 400

    if type(images) != list:
        images = [images]

    ratings = RaterNN.rateImageBatch(images)
    print(ratings)
    if user != "all":
        # filter out all ratings except for the user for each image
        for i in range(len(ratings)):
            ratings[i] = list(filter(lambda x: x[0] == user, ratings[i]))
            ratings[i] = ratings[i][0][1]
        return jsonify({"ratings": ratings}), 200
    else:
        # remove usernames from output
        # for i in range(len(ratings)):
        for i in range(len(ratings)):
            ratings[i] = list(map(lambda x: x[1], ratings[i]))
        return jsonify({"ratings": ratings, "users": RaterNN.usernames}), 200


@app.route("/addrating", methods=["POST"])
def app_addRating():
    image = request.files.get("image")
    if image is None:
        return jsonify({"error": "No image provided"}), 400
    username = request.form.get("user")
    rating = request.form.get("rating")
    user_and_rating = {
        "username": username,
        "rating": float(rating),
    }
    dataentry = Dataset.add_rating(
        image=image, user_and_rating=user_and_rating, RaterNN=RaterNN
    )
    Dataset.save_dataset("test/train.json")
    return jsonify(dataentry), 200

@app.route("/getuserdata", methods=["GET"])
def app_getUserData():
    username = request.form.get("user")
    if username is None:
        return jsonify({"error": "No user provided"}), 400
    if username not in RaterNN.usernames:
        return jsonify({"error": "Invalid user"}), 400
    userdata = Dataset.get_user_data(username)
    return jsonify(userdata), 200

@app.route("/getimage", methods=["GET"])
def app_getImage():
    image = request.form.get("image")
    if image is None:
        return jsonify({"error": "No image provided"}), 400
    image_data = Dataset.get_image(image)
    if image_data is None:
        return jsonify({"error": "Image not found"}), 400
    # send image data as an image
    response = app.response_class(
        response=image_data,
        status=200,
        mimetype="image/jpeg"
    )
    return response

def main():
    global config, TaggerNN, RaterNN, Dataset

    config = load_configs()
    TaggerNN, RaterNN = load_models(config, device=device)
    Dataset = RaterDataset(
        dataset_json="rater/train.json", imagefolder="rater/images", transform=None
    )

    CORS(app)
    app.run(host="0.0.0.0", port="2444")


if __name__ == "__main__":
    main()
