from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
from modules.utils import load_models, checkpoint_dataset_hash
from modules.raterdataset import RTData

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

app = Flask("AI Backend API Server")


@app.route("/tag", methods=["POST"])
def app_tag():
    # region Request validation

    image = request.files.get("image")
    if image is None:
        return jsonify({"error": "No image provided"}), 400

    # endregion

    tags = TaggerNN.tagImage(image)
    return jsonify({"tags": tags}), 200


@app.route("/tagbulk", methods=["POST"])
def app_tagBulk():
    # region Request validation

    images = request.files.getlist("images")
    if images is None:
        return jsonify({"error": "No images provided"}), 400

    # endregion

    if type(images) != list:
        images = [images]

    tags = TaggerNN.tagImageBatch(images)
    return jsonify({"tags": tags}), 200


@app.route("/rate", methods=["POST"])
def app_rate():
    # region Request validation

    image = request.files.get("image")
    user = request.form.get("user")
    if user is None:
        return jsonify({"error": "No user provided"}), 400
    if user not in RaterNN.usernames and user != "all":
        return jsonify({"error": "Invalid user"}), 400
    if image is None:
        return jsonify({"error": "No image provided"}), 400

    # endregion

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
    # region Request validation

    images = request.files.getlist("images")
    user = request.form.get("user")
    if user is None:
        user = "all"
    if user not in RaterNN.usernames and user != "all":
        return jsonify({"error": "Invalid user"}), 400
    if images is None:
        return jsonify({"error": "No images provided"}), 400

    # endregion

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
        for i in range(len(ratings)):
            ratings[i] = list(map(lambda x: x[1], ratings[i]))
        return jsonify({"ratings": ratings, "users": RaterNN.usernames}), 200


@app.route("/addrating", methods=["POST"])
def app_addRating():
    # region Request validation

    image = request.files.get("image")
    if image is None:
        return jsonify({"error": "No image provided"}), 400
    username = request.form.get("user")
    if username is None:
        return jsonify({"error": "No user provided"}), 400
    if username not in RaterNN.usernames:
        return jsonify({"error": "Invalid user"}), 400

    # endregion

    rating = float(request.form.get("rating"))
    dataentry = Tdata.add_rating(image=image, username=username, rating=rating)
    Tdata.save_dataset("rater/dataset.json")
    return jsonify(dataentry), 200


@app.route("/updaterating", methods=["POST"])
def app_updateRating():
    # region Request validation
    body = request.get_json()

    filename = body["filename"]
    if filename is None:
        return jsonify({"error": "No image provided"}), 400
    username = body["user"]
    if username is None:
        return jsonify({"error": "No user provided"}), 400
    if username not in RaterNN.usernames:
        return jsonify({"error": "Invalid user"}), 400
    rating = body["rating"]
    if rating is None:
        return jsonify({"error": "No rating provided"}), 400
    rating = float(rating)
    if rating < 0 or rating > 1:
        return jsonify({"error": "Invalid rating"}), 400

    # endregion

    dataentry = Tdata.update_rating(filename=filename, username=username, rating=rating)
    if dataentry is None:
        return jsonify({"error": "Image not found"}), 400
    Tdata.save_dataset("rater/dataset.json")
    return jsonify(dataentry), 200


@app.route("/getuserdata", methods=["GET"])
def app_getUserData():
    # region Request validation

    username = request.args.get("user")
    if username is None:
        return jsonify({"error": "No user provided"}), 400
    if username not in RaterNN.usernames:
        return jsonify({"error": "Invalid user"}), 400

    # endregion

    filters = request.args.get("filters")
    if filters is None:
        userdata = Tdata.get_userdataset(username).data
    else:
        filters = filters.split(",")
        filters = list(map(lambda x: x.strip(), filters))
        userdata = Tdata.get_userdataset_filtered(username, filters)
    page = request.args.get("page")
    limit = request.args.get("limit")
    if page is not None:
        page = int(page) - 1
        if limit is None:
            limit = 60
        else:
            limit = int(limit)
        max_page = len(userdata) // limit + 1
        userdata = userdata[page * limit : (page + 1) * limit]
    
    userdata = list(
        map(lambda x: {"image": x["image"], "rating": x["rating"]}, userdata)
    )
    if page is None:
        return jsonify({"images":userdata}), 200
    else:
        return jsonify({"images":userdata, "max_page":max_page}), 200

@app.route("/getimageneighbours", methods=["GET"])
def app_getImageNeighbours():

    # region Request validation

    filename = request.args.get("filename")
    if filename is None:
        return jsonify({"error": "No image filename provided"}), 400
    username = request.args.get("user")
    if username is None:
        return jsonify({"error": "No user provided"}), 400
    if username not in RaterNN.usernames:
        return jsonify({"error": "Invalid user"}), 400
    
    # endregion

    filters = request.args.get("filters")
    if filters is None:
        userdata = Tdata.get_userdataset(username).data
    else:
        filters = filters.split(",")
        filters = list(map(lambda x: x.strip(), filters))
        userdata = Tdata.get_userdataset_filtered(username, filters)

    # find index of filename image in userdata
    if filename not in list(map(lambda x: x["image"], userdata)):
        return jsonify({"error": "Image not found"}), 400

    index = list(map(lambda x: x["image"], userdata)).index(filename)

    response = {
        "position": index + 1,
        "max_image": len(userdata),
        "next_image": {"image": userdata[index + 1]["image"] if index < len(userdata) - 1 else None, "rating": userdata[index + 1]["rating"] if index < len(userdata) - 1 else None },
        "prev_image": {"image": userdata[index - 1]["image"] if index > 0 else None, "rating": userdata[index - 1]["rating"] if index > 0 else None },
    }

    return jsonify(response), 200  

@app.route("/getimage", methods=["GET"])
def app_getImage():
    # region Request validation

    filename = request.args.get("filename")
    if filename is None:
        return jsonify({"error": "No image filename provided"}), 400

    # endregion

    image = Tdata.get_image_2x(filename)
    if image is None:
        return jsonify({"error": "Image not found"}), 400
    response = app.response_class(response=image, status=200, mimetype="image/jpeg")
    return response


@app.route("/getimagetags", methods=["GET"])
def app_getImageTags():
    # region Request validation

    filename = request.args.get("filename")
    if filename is None:
        return jsonify({"error": "No image filename provided"}), 400

    # endregion

    tags = Tdata.get_image_tags(filename)
    if tags is None:
        return jsonify({"error": "Image not found"}), 400
    return jsonify({"tags": tags}), 200

@app.route("/getstats", methods=["GET"])    
def app_getStats():
    # region Request validation

    username = request.args.get("user")
    if username is None:
        return jsonify({"error": "No user provided"}), 400
    if username not in RaterNN.usernames:
        return jsonify({"error": "Invalid user"}), 400

    # endregion

    stats = Tdata.get_userdataset(username).get_stats(username)
    if current_training != "None":
        trainerstats = current_training.get_status()
        stats["trainer"] = trainerstats
    else:
        stats["trainer"] = None

    return jsonify(stats), 200


@app.route("/verifyfulldataset", methods=["GET"])
def app_verifyDatasets():
    valid = Tdata.verify_full_dataset()
    return jsonify({"valid": valid}), 200

@app.route("/createfulldataset", methods=["GET"])
def app_createFullDataset():
    valid = Tdata.verify_full_dataset()
    if valid:
        return jsonify({"error": "Full dataset already up-to-date"}), 400
    retraining_needed = Tdata.pre_verify_usersets_for_full_dataset()
    if retraining_needed is not None:
        return jsonify({"error": "Retraining needed", "users": retraining_needed}), 400
    Tdata.create_full_dataset()
    Tdata.save_dataset("rater/dataset.json")
    return jsonify({"success": True}), 200

@app.route("/updatetags", methods=["GET"])
def app_updateTags():
    Tdata.update_tags(tagger=TaggerNN)
    Tdata.save_dataset("rater/dataset.json")
    return jsonify({"success": True}), 200


@app.route("/trainuser", methods=["GET"])
def app_trainUser():
    # region Request validation
    username = request.args.get("user")
    if username is None:
        return jsonify({"error": "No user provided"}), 400
    if username not in RaterNN.usernames:
        return jsonify({"error": "Invalid user"}), 400

    # endregion

    global current_training

    if current_training != "None" and current_training.is_training():
        return jsonify({"error": "Already training"}), 400

    # region Check if model is already trained
    modelhash = checkpoint_dataset_hash("models/RaterNNP_" + username + ".pth")
    dataset_hash = Tdata.get_userdataset(username).dataset_hash

    if modelhash == dataset_hash:
        return jsonify({"error": "Model already trained"}), 400
    # endregion

    from modules.training import PTrainer

    current_training = PTrainer(username=username, tdata=Tdata)
    current_training.start_training()

    return jsonify({"success": True}), 200

@app.route("/trainall", methods=["GET"])
def app_trainAll():
    global current_training

    if current_training != "None" and current_training.is_training():
        return jsonify({"error": "Already training"}), 400

    # region Check if model is already trained
    modelhash = checkpoint_dataset_hash("models/RaterNN.pth")
    dataset_hash = Tdata.dataset_hash

    ## TODO: Check dates of files

    if modelhash == dataset_hash:
        return jsonify({"error": "Model already trained"}), 400
    # endregion

    valid = Tdata.verify_full_dataset()
    if not valid:
        return jsonify({"error": "Full dataset is not up-to-date"}), 400

    from modules.training import Trainer
    global RaterNN

    current_training = Trainer(tdata=Tdata)
    current_training.start_training(raternn=RaterNN)

    return jsonify({"success": True}), 200

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

@app.route("/removeduplicates", methods=["GET"])
def app_removeDuplicates():
    Tdata.remove_duplicates()
    Tdata.save_dataset("rater/dataset.json")
    return jsonify({"success": True}), 200

def main():
    global TaggerNN, RaterNN, Tdata, current_training

    TaggerNN, RaterNN = load_models(device=device)
    from modules.utils import get_val_transforms

    current_training = "None"

    Tdata = RTData(dataset_json="rater/dataset.json", transform=get_val_transforms())
    if Tdata.update_needed:
        print("Updating tags...", flush=True, end="")
        Tdata.update_tags(tagger=TaggerNN)
        Tdata.save_dataset("rater/dataset.json")
        print("Done!", flush=True)

    CORS(app)
    app.run(host="0.0.0.0", port=2444)


if __name__ == "__main__":
    main()
