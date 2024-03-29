from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
from modules.utils import load_models, checkpoint_dataset_hash
from modules.datasets import RTData
import modules.db_functions as dbf

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
    # TODO: change this to use discord_ids instead of usernames

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

    if user != "all":
        # filter out all ratings except for the user
        ratings = list(filter(lambda x: x[0] == user, ratings))
        ratings = ratings[0][1]
        return jsonify(ratings), 200
    else:
        return jsonify({"ratings": ratings}), 200


@app.route("/ratebulk", methods=["POST"])
def app_rateBulk():
    # TODO: change this to use discord_ids instead of usernames

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
    try:
        discord_id = int(request.form.get("user"))
    except:
        return jsonify({"error": "Invalid user"}), 400
    if discord_id not in dbf.get_discord_ids():
        return jsonify({"error": "Invalid user"}), 400

    rating = request.form.get("rating")
    if rating is None:
        return jsonify({"error": "No rating provided"}), 400
    
    rating = float(rating)

    if rating < 0 or rating > 1:
        return jsonify({"error": "Invalid rating"}), 400
    
    # endregion

    success = dbf.add_rating(image, discord_id, rating)
    
    if not success:
        return jsonify({"error": "Filename already in the database"}), 400
    print("rating added")

    # we need to update tags in the database, because the new image doesn't have any tags yet
    global TaggerNN
    from modules.utils import update_tags
    print("updating tags")
    updated_images_count = update_tags(taggernn=TaggerNN)
    print("updated tags")

    if updated_images_count < 1:
        print("Failed to update tags")
        return jsonify({"error": "Failed to update tags"}), 500

    return jsonify({"success": True}), 200


@app.route("/updaterating", methods=["POST"])
def app_updateRating():
    # region Request validation
    body = request.get_json()

    filename = body["filename"]
    if filename is None:
        return jsonify({"error": "No image provided"}), 400
    try:
        discord_id = int(body["user"])
    except:
        return jsonify({"error": "Invalid user"}), 400
    if discord_id not in dbf.get_discord_ids():
        return jsonify({"error": "Invalid user"}), 400
    rating = body["rating"]
    if rating is None:
        return jsonify({"error": "No rating provided"}), 400
    rating = float(rating)
    if rating < 0 or rating > 1:
        return jsonify({"error": "Invalid rating"}), 400

    # endregion

    success = dbf.update_rating(filename, discord_id, rating)
    if not success:
        return jsonify({"error": "Image not found"}), 400
    
    return jsonify({"success": True}), 200

@app.route("/removerating", methods=["GET"])
def app_removeRating():
    # region Request validation

    filename = request.args.get("filename")
    if filename is None:
        return jsonify({"error": "No image provided"}), 400
    try:
        discord_id = int(request.args.get("user"))
    except:
        return jsonify({"error": "Invalid user"}), 400
    if discord_id not in dbf.get_discord_ids():
        return jsonify({"error": "Invalid user"}), 400

    # endregion

    success = dbf.remove_rating(filename, discord_id)
    if not success:
        return jsonify({"error": "Image not found"}), 400
    
    return jsonify({"success": True}), 200

@app.route("/deleteimage", methods=["GET"])
def app_deleteImage():
    # region Request validation

    filename = request.args.get("filename")
    if filename is None:
        return jsonify({"error": "No image provided"}), 400

    # endregion

    success = dbf.delete_image(filename)
    if not success:
        return jsonify({"error": "Image not found"}), 400
    
    return jsonify({"success": True}), 200

@app.route("/getuserdata", methods=["GET"])
def app_getUserData():

    # region Request validation

    try:
        discord_id = int(request.args.get("user"))
    except:
        return jsonify({"error": "Invalid user"}), 400
    if discord_id not in dbf.get_discord_ids():
        return jsonify({"error": "Invalid user"}), 400

    rated = request.args.get("rated")
    if rated is None:
        rated = "yes"
    elif rated not in ["yes", "no", "all"]:
        return jsonify({"error": "Invalid rated parameter"}), 400
    
    sort = request.args.get("sort")
    if sort is None:
        sort = "date-desc"

    filters = request.args.get("filters")
    if filters is not None:
        filters = filters.split(",")
        filters = list(map(lambda x: x.strip(), filters))

    # endregion

    userdata = dbf.get_data(discord_id, rated, sort, filters)
        
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
    try:
        discord_id = int(request.args.get("user"))
    except:
        return jsonify({"error": "Invalid user"}), 400
    if discord_id not in dbf.get_discord_ids():
        return jsonify({"error": "Invalid user"}), 400
    
    rated = request.args.get("rated")
    if rated is None:
        rated = "yes"
    elif rated not in ["yes", "no", "all"]:
        return jsonify({"error": "Invalid rated parameter"}), 400
    
    sort = request.args.get("sort")
    if sort is None:
        sort = "date-desc"

    filters = request.args.get("filters")
    if filters is not None:
        filters = filters.split(",")
        filters = list(map(lambda x: x.strip(), filters))

    # endregion
    
    userdata = dbf.get_data(discord_id, rated, sort, filters)

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

    mode = request.args.get("mode")
    if mode is None:
        mode = "768"

    # endregion

    imagefile = dbf.get_image(filename, mode)
    if imagefile is None:
        return jsonify({"error": "Image not found"}), 400
    response = app.response_class(response=imagefile.read(), status=200, mimetype="image/jpeg")
    return response

@app.route("/addimage", methods=["POST"])
def app_addImage():
    # region Request validation

    image = request.files.get("image")
    if image is None:
        return jsonify({"error": "No image provided"}), 400

    sankaku_id = request.form.get("sankaku_id")
    # endregion

    filename = dbf.add_image(image, sankaku_id if sankaku_id is not None else None)
    if not filename:
        return jsonify({"error": "Filename already in the database"}), 400
    
    return jsonify({"success": True, "filename": filename}), 200

@app.route("/getthumbnail", methods=["GET"])
def app_getThumbnail():
    # region Request validation

    filename = request.args.get("filename")
    if filename is None:
        return jsonify({"error": "No image filename provided"}), 400

    # endregion

    imagefile = dbf.get_thumbnail_image(filename)
    if imagefile is None:
        return jsonify({"error": "Image not found"}), 400
    response = app.response_class(response=imagefile.read(), status=200, mimetype="image/webp")
    return response

@app.route("/getimagetags", methods=["GET"])
def app_getImageTags():
    # region Request validation

    filename = request.args.get("filename")
    if filename is None:
        return jsonify({"error": "No image filename provided"}), 400

    # endregion

    tags = dbf.get_image_tags(filename)
    if tags is None:
        return jsonify({"error": "Image not found"}), 400
    return jsonify({"tags": tags}), 200

@app.route("/getstats", methods=["GET"])    
def app_getStats():
    # region Request validation

    try:
        discord_id = int(request.args.get("user"))
    except:
        return jsonify({"error": "Invalid user"}), 400
    if discord_id not in dbf.get_discord_ids():
        return jsonify({"error": "Invalid user"}), 400

    # endregion

    stats = dbf.get_dataset_stats(discord_id)
    if current_training != "None":
        trainerstats = current_training.get_status()
        stats["trainer"] = trainerstats
    else:
        stats["trainer"] = None

    return jsonify(stats), 200

@app.route("/gettags", methods=["GET"])
def app_getTags():
    tags = dbf.get_tags()
    return jsonify({"tags": tags}), 200

# Montageposts

@app.route("/getmontageposts", methods=["GET"])
def app_getMontageposts():
    # region Request validation

    try:
        discord_id = int(request.args.get("user"))
    except:
        return jsonify({"error": "Invalid user"}), 400
    if discord_id not in dbf.get_discord_ids():
        return jsonify({"error": "Invalid user"}), 400
    
    filters = request.args.get("filters")
    if filters is not None:
        filters = filters.split(",")
        filters = list(map(lambda x: x.strip(), filters))

    # endregion

    montageposts = dbf.get_montageposts(discord_id, filters)

    page = request.args.get("page")
    limit = request.args.get("limit")
    if page is not None:
        page = int(page) - 1
        if limit is None:
            limit = 15
        else:
            limit = int(limit)
        max_page = len(montageposts) // limit + 1
        montageposts = montageposts[page * limit : (page + 1) * limit]

    if page is None:
        return jsonify({"montageposts": montageposts}), 200
    else:
        return jsonify({"montageposts": montageposts, "max_page": max_page}), 200

@app.route("/getmontagepostneighbours", methods=["GET"])
def app_getMontagepostNeighbours():
    # region Request validation

    montagepost_id = request.args.get("id")
    if montagepost_id is None:
        return jsonify({"error": "No montagepost ID provided"}), 400
    try:
        montagepost_id = int(montagepost_id)
    except:
        return jsonify({"error": "Invalid montagepost ID"}), 400

    try:
        discord_id = int(request.args.get("user"))
    except:
        return jsonify({"error": "Invalid user"}), 400
    if discord_id not in dbf.get_discord_ids():
        return jsonify({"error": "Invalid user"}), 400
    
    filters = request.args.get("filters")
    if filters is not None:
        filters = filters.split(",")
        filters = list(map(lambda x: x.strip(), filters))

    # endregion

    montageposts = dbf.get_montageposts(discord_id, filters)

    # find index of filename image in userdata
    if montagepost_id not in list(map(lambda x: x["id"], montageposts)):
        return jsonify({"error": "Montagepost not found"}), 400

    index = list(map(lambda x: x["id"], montageposts)).index(montagepost_id)

    response = {
        "position": index + 1,
        "max_montagepost": len(montageposts),
        "next_montagepost": {"id": montageposts[index + 1]["id"] if index < len(montageposts) - 1 else None, "created_at": montageposts[index + 1]["created_at"] if index < len(montageposts) - 1 else None },
        "prev_montagepost": {"id": montageposts[index - 1]["id"] if index > 0 else None, "created_at": montageposts[index - 1]["created_at"] if index > 0 else None },
    }

    return jsonify(response), 200

@app.route("/getmontagepost", methods=["GET"])
def app_getMontagepost():
    # region Request validation

    try:
        montagepost_id = int(request.args.get("id"))
    except:
        return jsonify({"error": "Invalid montagepost ID"}), 400

    # endregion

    montagepost = dbf.get_montagepost(montagepost_id)
    return jsonify({"montagepost": montagepost}), 200

@app.route("/createmontagepost", methods=["POST"])
def app_createMontagepost():
    # region Request validation

    try:
        discord_id = int(request.form.get("user"))
    except:
        return jsonify({"error": "Invalid user"}), 400
    
    if discord_id not in dbf.get_discord_ids():
        return jsonify({"error": "Invalid user"}), 400
   
    filenames = request.form.get("filenames").split(",")
    if filenames is None:
        return jsonify({"error": "No filenames provided"}), 400
    # endregion

    montagepost_id = dbf.create_montagepost(filenames=filenames, discord_id=discord_id)
    return jsonify({"montagepost_id": montagepost_id}), 200

# Dataset

# TODO: Implement this using the database
@app.route("/verifyfulldataset", methods=["GET"])
def app_verifyDatasets():
    valid = Tdata.verify_full_dataset()
    return jsonify({"valid": valid}), 200

# TODO: Implement this using the database
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
    full = False
    if request.args.get("full"):
        full = request.args.get("full") == "true"

    global TaggerNN
    from modules.utils import update_tags
    updated_images_count = update_tags(taggernn=TaggerNN, full=full)

    # clean cuda cache
    torch.cuda.empty_cache()

    return jsonify({"updated_images_count": updated_images_count}), 200


# Training

@app.route("/trainuser", methods=["GET"])
def app_trainUser():
    # region Request validation
    try:
        discord_id = int(request.args.get("user"))
    except:
        return jsonify({"error": "Invalid user"}), 400
    if discord_id not in dbf.get_discord_ids():
        return jsonify({"error": "Invalid user"}), 400

    # endregion

    global current_training

    if current_training != "None" and current_training.is_training():
        return jsonify({"error": "Already training"}), 400

    # region Check if model is already trained
    modelhash = checkpoint_dataset_hash(f"models/RaterNNP_{discord_id}.pth")
    dataset = dbf.create_RPDataset(discord_id)
    dataset_hash = dataset.generate_hash()

    if modelhash == dataset_hash:
        return jsonify({"error": "Model already trained"}), 400
    # endregion

    from modules.training import PTrainer

    current_training = PTrainer(dataset=dataset)
    current_training.start_training()

    return jsonify({"success": True}), 200

# TODO: Implement this using the database
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

# TODO: Implement this using the database
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
    global TaggerNN, RaterNN, Tdata, current_training

    TaggerNN, RaterNN = load_models(device=device)

    # clean cuda cache
    torch.cuda.empty_cache()

    from modules.utils import get_val_transforms

    current_training = "None"

    CORS(app)
    app.run(host="0.0.0.0", port=2444)


if __name__ == "__main__":
    main()
