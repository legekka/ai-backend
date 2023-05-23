from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
from modules.utils import load_configs, load_models
from modules.raterdataset import RTData

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

app = Flask("AI Backend API Server")


@app.route("/tag", methods=["POST"])
def app_tag():
    image = request.files.get("image")
    if image is None:
        return jsonify({"error": "No image provided"}), 400
    tags = TaggerNN.tagImage(image)
    return jsonify({"tags": tags}), 200

@app.route("/tagbulk", methods=["POST"])
def app_tagBulk():
    images = request.files.getlist("images")
    if images is None:
        return jsonify({"error": "No images provided"}), 400

    if type(images) != list:
        images = [images]

    tags = TaggerNN.tagImageBatch(images)
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
    rating = float(request.form.get("rating"))
    dataentry = Tdata.add_rating(image=image, username=username, rating=rating)
    Tdata.save_dataset("rater/dataset.json")
    return jsonify(dataentry), 200


@app.route("/getuserdata", methods=["GET"])
def app_getUserData():
    username = request.args.get("user")
    if username is None:
        return jsonify({"error": "No user provided"}), 400
    if username not in RaterNN.usernames:
        return jsonify({"error": "Invalid user"}), 400
    userdata = Tdata.get_userdataset(username).data
    return jsonify(userdata), 200


@app.route("/getimage", methods=["GET"])
def app_getImage():
    filename = request.args.get("filename")
    if filename is None:
        return jsonify({"error": "No image filename provided"}), 400
    image = Tdata.get_image_2x(filename)
    if image is None:
        return jsonify({"error": "Image not found"}), 400
    response = app.response_class(response=image, status=200, mimetype="image/jpeg")
    return response

@app.route("/getimagetags", methods=["GET"])
def app_getImageTags():
    filename = request.args.get("filename")
    if filename is None:
        return jsonify({"error": "No image filename provided"}), 400
    tags = Tdata.get_image_tags(filename)
    if tags is None:
        return jsonify({"error": "Image not found"}), 400
    return jsonify({"tags": tags}), 200


@app.route("/verifydatasets", methods=["GET"])
def app_verifyDatasets():
    valid = Tdata.verify_full_dataset()
    return jsonify({"valid": valid}), 200

@app.route("/updatetags", methods=["GET"])
def app_updateTags():
    Tdata.update_tags(tagger=TaggerNN)
    Tdata.save_dataset("rater/dataset.json")
    return jsonify({"success": True}), 200

@app.route("/trainuser", methods=["POST"])
def app_trainUser():
    username = request.args.get("user")
    if username is None:
        return jsonify({"error": "No user provided"}), 400
    if username not in RaterNN.usernames:
        return jsonify({"error": "Invalid user"}), 400

    global current_training

    if current_training == "None" or not current_training.is_training():
        from modules.training import PTrainer

        current_training = PTrainer(username=username, tdata=Tdata)
        current_training.start_training()

        return jsonify({"success": True}), 200
    else:
        return jsonify({"error": "Already training"}), 400

@app.route("/stoptraining", methods=["GET"])
def app_stopTraining():
    global current_training
    if current_training == "None" or not current_training.is_training():
        return jsonify({"error": "Not training"}), 400
    else:
        current_training.stop_training()
        return jsonify({"success": True}), 200

@app.route("/trainerstatus", methods=["GET"])
def app_trainerStatus():
    if current_training == "None":
        return jsonify({"status": {"is_training": False}}), 200
    else:
        return jsonify({"status": current_training.get_status()}), 200


def main():
    global config, TaggerNN, RaterNN, Tdata, current_training
    config = load_configs()
    TaggerNN, RaterNN = load_models(config, device=device)
    from modules.utils import get_val_transforms

    current_training = "None"

    Tdata = RTData(dataset_json="rater/dataset.json", transform=get_val_transforms())
    CORS(app)
    app.run(host="0.0.0.0", port=2444)


if __name__ == "__main__":
    main()
