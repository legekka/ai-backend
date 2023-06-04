from modules.db_model import *

def _get_order_by_param(sort):
    # this is a helper function for get_userdata and get_userdata_filtered
    match sort:
        case "date-asc":
            order_by_param = Image.id
        case "date-desc":
            order_by_param = Image.id.desc()
        case "rating-asc":
            order_by_param = Rating.rating
        case "rating-desc":
            order_by_param = Rating.rating.desc()
        case "filename-asc":
            order_by_param = Image.filename
        case "filename-desc":
            order_by_param = Image.filename.desc()
        case _:
            order_by_param = Image.id.desc()

    return order_by_param

def get_users():
    usernames = (User
        .select(User.username)
        .dicts()
        )
    
    formatted_usernames = []
    for username in usernames:
        formatted_usernames.append(username["username"])
    
    return formatted_usernames

def get_userdata(username, sort=None):

    ratings = (Rating
        .select(Image.filename, Rating.rating, Image.id)
        .join(Image)
        .switch(Rating)
        .join(User)
        .where(User.username == username)
        .order_by(_get_order_by_param(sort))
        .dicts()
        )
    
    formatted_ratings = []

    for rating in ratings:
        formatted_ratings.append({
            "image": rating["filename"],
            "rating": float(rating["rating"])
        })
    
    return formatted_ratings

def get_userdata_filtered(username, filters, sort=None):
    # this is similar to get_userdata, but also connects the images to the tags table, to check whether the tags in the filters dict are present in the image tags
    # Connections:
    #  Rating-->Image
    #  Rating-->User
    #  Image-->ImageTag
    #  ImageTag-->Tag
    ratings = (Rating
        .select(Image.filename, Rating.rating, Image.id)
        .join(Image)
        .switch(Rating)
        .join(User)
        .switch(Image)
        .join(ImageTag)
        .switch(ImageTag)
        .join(Tag)
        .where(User.username == username, Tag.name.in_(filters))
        # we have to check if the count of the tags is equal to the count of the filters, because if we don't, we'll get images that have at least one of the tags, not all of them
        .group_by(Image.filename, Rating.rating, Image.id)
        .having(fn.COUNT(Tag.name) == len(filters))
        .order_by(_get_order_by_param(sort)) 
        .dicts()
        )
    
    formatted_ratings = []
    for rating in ratings:
        formatted_ratings.append({
            "image": rating["filename"],
            "rating": float(rating["rating"])
        })
    
    return formatted_ratings

def get_image(filename):
    # this just gets the image data from the database
    imagefile = (Image
        .select(Image.image_768)
        .where(Image.filename == filename)
        .dicts()
        )[0]["image_768"]
    
    if imagefile == None:
        return None
    
    import io
    import base64
    # this is a base64 encoded and decoded string, so we need to convert it to bytes
    imagefile = base64.b64decode(imagefile)
    fileobject = io.BytesIO()
    fileobject.write(imagefile)
    fileobject.seek(0)

    return fileobject.read()

def get_image_tags(filename):
    # this as it sounds, gets the tags for an image
    tags = (ImageTag
        .select(Tag.name)
        .join(Tag)
        .switch(ImageTag)
        .join(Image)
        .where(Image.filename == filename)
        .order_by(Tag.name)
        .dicts()
        )
    
    formatted_tags = list(map(lambda x: x["name"], tags))
    
    return formatted_tags

def generate_dataset_hash(username):
    data = get_userdata(username)

    import hashlib
    import json

    return hashlib.sha256(json.dumps(data).encode()).hexdigest()

def update_rating(filename, username, rating_value):
    # first find the rating
    rating = (Rating
        .select()
        .join(Image)
        .switch(Rating)
        .join(User)
        .where(Image.filename == filename, User.username == username)
        )
    
    # if the rating exists, update it
    if len(rating) > 0:
        from modules.utils import align_rating

        try:
            rating = rating[0]
            rating.rating = align_rating(rating_value)
            rating.save()
        except Exception as e:
            print("Error updating rating: " + str(e))
            return False
        return True
    
    return False

def add_rating(filename, username, rating_value):
    # TODO: implement
    pass

def get_dataset_stats(username):
    data = get_userdata(username)

    summary = [0, 0, 0, 0, 0, 0, 0]
    for i in range(len(data)):
        rating = round(data[i]["rating"] * 6)
        summary[int(rating)] += 1

    from modules.utils import checkpoint_dataset_hash, raternn_up_to_date

    raternnp_hash = checkpoint_dataset_hash("models/RaterNNP_" + username + ".pth")
    dataset_hash = generate_dataset_hash(username)

    stats = {
        "image_count": len(data),
        "rating_distribution": summary,
        "RaterNNP_up_to_date": raternnp_hash == dataset_hash,
        "RaterNN_up_to_date": raternn_up_to_date(get_users()),
    }
    return stats
