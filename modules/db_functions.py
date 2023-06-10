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


def get_discord_ids():
    discord_ids = User.select(User.discord_id).dicts()

    formatted_discord_ids = []
    for discord_id in discord_ids:
        formatted_discord_ids.append(discord_id["discord_id"])

    return formatted_discord_ids


def get_usernames():
    usernames = User.select(User.username).dicts()

    formatted_usernames = []
    for username in usernames:
        formatted_usernames.append(username["username"])

    return formatted_usernames


def get_userdata(discord_id, sort=None):
    ratings = (
        Rating.select(Image.filename, Rating.rating, Image.id)
        .join(Image)
        .switch(Rating)
        .join(User)
        .where(User.discord_id == discord_id)
        .order_by(_get_order_by_param(sort))
        .dicts()
    )

    formatted_ratings = []

    for rating in ratings:
        formatted_ratings.append(
            {"image": rating["filename"], "rating": float(rating["rating"])}
        )

    return formatted_ratings


def get_userdata_filtered(discord_id, filters, sort=None):
    # this is similar to get_userdata, but also connects the images to the tags table, to check whether the tags in the filters dict are present in the image tags
    ratings = (
        Rating.select(Image.filename, Rating.rating, Image.id)
        .join(Image)
        .switch(Rating)
        .join(User)
        .switch(Image)
        .join(ImageTag)
        .switch(ImageTag)
        .join(Tag)
        .where(User.discord_id == discord_id, Tag.name.in_(filters))
        .group_by(Image.filename, Rating.rating, Image.id)
        .having(fn.COUNT(Tag.name) == len(filters))
        .order_by(_get_order_by_param(sort))
        .dicts()
    )

    formatted_ratings = []
    for rating in ratings:
        formatted_ratings.append(
            {"image": rating["filename"], "rating": float(rating["rating"])}
        )

    return formatted_ratings


def get_data(discord_id, rated, sort=None, filters=None):
    user_id = (User.select(User.id).where(User.discord_id == discord_id).dicts())[0][
        "id"
    ]

    # depending on datatype, it gives back images that the user has rated, or images that the user has not rated
    if rated == "yes":
        # if rated, then we will just use get_userdata or get_userdata_filtered
        if filters == None:
            return get_userdata(discord_id, sort)
        else:
            return get_userdata_filtered(discord_id, filters, sort)
    elif rated == "no":
        # if unrated, then we will get all images, and then filter out the ones that the user has rated
        # first we need to get the user_id
        if sort in ["rating-asc", "rating-desc"]:
            sort = "date-desc"

        if filters == None:
            images = (
                Image.select(Image.filename, Image.id)
                .join(Rating, JOIN.LEFT_OUTER)
                .where(Rating.user_id != user_id)
                .group_by(Image.filename, Image.id)
                .order_by(_get_order_by_param(sort))
                .dicts()
            )
        else:
            images = (
                Image.select(Image.filename, Image.id)
                .join(Rating, JOIN.LEFT_OUTER)
                .switch(Image)
                .join(ImageTag)
                .switch(ImageTag)
                .join(Tag)
                .where(Rating.user_id != user_id, Tag.name.in_(filters))
                .group_by(Image.filename, Image.id)
                .having(fn.COUNT(Tag.name) == len(filters))
                .order_by(_get_order_by_param(sort))
                .dicts()
            )

        formatted_images = []
        for image in images:
            formatted_images.append({"image": image["filename"], "rating": float(-1)})

        return formatted_images
    elif rated == "all":
        if filters == None:
            # this is a bit tricky, we need to make one query, because it would be too slow otherwise
            # so the approach is, we will the images, connected to the ratings, and we will try to group them by the image id, but keeping the user rating.
            # then we will check the user_ids in the ratings, and if it's not the user_id, we will set the rating to -1

            images_rated = (
                Image.select(Image.filename, Rating.rating, Image.id)
                .join(Rating, JOIN.LEFT_OUTER)
                .where(Rating.user_id == user_id)
                .dicts()
            )

            # this is how using the ORM the query above would look like
            images_not_rated = (
                Image.select(Image.filename, Image.id)
                .join(Rating, JOIN.LEFT_OUTER)
                .where(
                    Image.filename.not_in(
                        Image.select(Image.filename)
                        .join(Rating, JOIN.LEFT_OUTER)
                        .where(Rating.user_id == user_id)
                        .group_by(Image.filename)
                    )
                )
                .group_by(Image.filename, Image.id)
                .dicts()
            )
        else:
            # this is now the same as above but with filters
            images_rated = (
                Image.select(Image.filename, Rating.rating, Image.id)
                .join(Rating, JOIN.LEFT_OUTER)
                .switch(Image)
                .join(ImageTag)
                .switch(ImageTag)
                .join(Tag)
                .where(Rating.user_id == user_id, Tag.name.in_(filters))
                .group_by(Image.filename, Rating.rating, Image.id)
                .having(fn.COUNT(Tag.name) == len(filters))
                .dicts()
            )

            images_not_rated = (
                Image.select(Image.filename, Image.id)
                .join(Rating, JOIN.LEFT_OUTER)
                .switch(Image)
                .join(ImageTag)
                .switch(ImageTag)
                .join(Tag)
                .where(
                    Image.filename.not_in(
                        Image.select(Image.filename)
                        .join(Rating, JOIN.LEFT_OUTER)
                        .where(Rating.user_id == user_id)
                        .group_by(Image.filename)
                    )
                )
                .where(Tag.name.in_(filters))
                .group_by(Image.filename, Image.id)
                .having(fn.COUNT(Tag.name) == len(filters))
                .dicts()
            )

        formatted_images_rated = []
        for image in images_rated:
            formatted_images_rated.append(
                {
                    "id": image["id"],
                    "image": image["filename"],
                    "rating": float(image["rating"]),
                }
            )

        formatted_images_not_rated = []
        for image in images_not_rated:
            formatted_images_not_rated.append(
                {"id": image["id"], "image": image["filename"], "rating": float(-1)}
            )

        formatted_images = formatted_images_rated + formatted_images_not_rated

        match sort:
            case "date-asc":
                formatted_images.sort(key=lambda x: x["id"])
            case "date-desc":
                formatted_images.sort(key=lambda x: x["id"], reverse=True)
            case "rating-asc":
                formatted_images.sort(key=lambda x: x["rating"])
            case "rating-desc":
                formatted_images.sort(key=lambda x: x["rating"], reverse=True)
            case "filename-asc":
                formatted_images.sort(key=lambda x: x["image"])
            case "filename-desc":
                formatted_images.sort(key=lambda x: x["image"], reverse=True)
            case _:
                formatted_images.sort(key=lambda x: x["id"])

        # remove id from formatted_images
        for image in formatted_images:
            del image["id"]

        return formatted_images


def get_image(filename):
    # this just gets the image data from the database
    imagefile = (
        Image.select(Image.image_768).where(Image.filename == filename).dicts()
    )[0]["image_768"]

    if imagefile == None:
        return None

    import io
    import base64

    imagefile = base64.b64decode(imagefile)
    fileobject = io.BytesIO()
    fileobject.write(imagefile)
    fileobject.seek(0)

    return fileobject

def get_thumbnail_image(filename):
    # this just gets the image data from the database
    imagefile = (
        Image.select(Image.image_305).where(Image.filename == filename).dicts()
    )[0]["image_305"]

    if imagefile == None:
        return None

    import io
    import base64

    imagefile = base64.b64decode(imagefile)
    fileobject = io.BytesIO()
    fileobject.write(imagefile)
    fileobject.seek(0)

    return fileobject

def get_images(filenames, mode="768"):
    # this gets multiple images from the database
    if mode == "768":
        images = (
            Image.select(Image.filename, Image.image_768)
            .where(Image.filename.in_(filenames))
            .dicts()
        )
    else:
        images = (
            Image.select(Image.filename, Image.image_512_t)
            .where(Image.filename.in_(filenames))
            .dicts()
        )

    imagelist = []
    for image in images:
        import io
        import base64

        if mode == "768":
            imagefile = base64.b64decode(image["image_768"])
        else:
            imagefile = base64.b64decode(image["image_512_t"])
        fileobject = io.BytesIO()
        fileobject.write(imagefile)
        fileobject.seek(0)

        imagelist.append(fileobject)

    return imagelist


def get_image_tags(filename):
    # this as it sounds, gets the tags for an image
    tags = (
        ImageTag.select(Tag.name)
        .join(Tag)
        .switch(ImageTag)
        .join(Image)
        .where(Image.filename == filename)
        .order_by(Tag.name)
        .dicts()
    )

    formatted_tags = list(map(lambda x: x["name"], tags))

    return formatted_tags


def get_all_image_filenames():
    # this gets all image filenames
    images = Image.select(Image.filename).dicts()

    formatted_images = []
    for image in images:
        formatted_images.append(image["filename"])

    return formatted_images


def get_image_filenames_with_missing_tags():
    # this gets all images that have no tags
    images = (
        Image.select(Image.filename)
        .join(ImageTag, JOIN.LEFT_OUTER)
        .where(ImageTag.id.is_null())
        .dicts()
    )

    formatted_images = []
    for image in images:
        formatted_images.append(image["filename"])

    return formatted_images


def generate_dataset_hash(discord_id):
    data = create_RPDataset(discord_id)

    import hashlib
    import json

    return hashlib.sha256(json.dumps(data).encode()).hexdigest()


def update_tags(filename, tags):
    # this updates the tags for an image
    # first, we need to get the image.id, then we delete the old tags from ImageTags

    image_id = (Image.select(Image.id).where(Image.filename == filename).dicts())[0][
        "id"
    ]

    (ImageTag.delete().where(ImageTag.image_id == image_id).execute())

    # then, add the new tags
    for tag in tags:
        tag_id = (Tag.select(Tag.id).where(Tag.name == tag).dicts())[0]["id"]

        ImageTag.create(image_id=image_id, tag_id=tag_id)

    return True


def update_rating(filename, discord_id, rating_value):
    # first find the rating
    rating = (
        Rating.select()
        .join(Image)
        .switch(Rating)
        .join(User)
        .where(Image.filename == filename, User.discord_id == discord_id)
    )

    from modules.utils import align_rating

    # if the rating exists, update it
    if len(rating) > 0:
        try:
            rating = rating[0]
            rating.rating = align_rating(rating_value)
            rating.save()
        except Exception as e:
            print("Error updating rating: " + str(e))
            return False
        return True
    else:
        # otherwise, create it
        try:
            image_id = (
                Image.select(Image.id).where(Image.filename == filename).dicts()
            )[0]["id"]
        except:
            print("Error getting image_id")
            return False

        user_id = (User.select(User.id).where(User.discord_id == discord_id).dicts())[
            0
        ]["id"]

        try:
            Rating.create(
                image_id=image_id, user_id=user_id, rating=align_rating(rating_value)
            )
        except Exception as e:
            print("Error creating rating: " + str(e))
            return False

        return True


def add_rating(imageobject, discord_id, rating_value):
    # first of all, we have to check the image.filename in the database

    image = Image.select().where(Image.filename == imageobject.filename).dicts()

    if len(image) != 0:
        # if the image exists, we won't add it
        return False

    # preparing the image for database storage

    from modules.utils import convert_to_image_512_t, convert_to_image_768, convert_to_image_305, align_rating
    import base64

    image_512_t = convert_to_image_512_t(imageobject).read()
    image_768 = convert_to_image_768(imageobject).read()
    image_305 = convert_to_image_305(imageobject).read()

    image_512_t = base64.b64encode(image_512_t).decode("utf-8")
    image_768 = base64.b64encode(image_768).decode("utf-8")
    image_305 = base64.b64encode(image_305).decode("utf-8")

    filename = imageobject.filename

    # now we can add the image to the database
    try:
        Image.create(filename=filename, image_512_t=image_512_t, image_768=image_768, image_305=image_305)
    except Exception as e:
        print("Error adding image to database: " + str(e))
        return False

    # and now we will add the rating. We will get the image_id and user_id tho
    image_id = (Image.select(Image.id).where(Image.filename == filename).dicts())[0][
        "id"
    ]

    user_id = (User.select(User.id).where(User.discord_id == discord_id).dicts())[0][
        "id"
    ]

    try:
        Rating.create(
            image_id=image_id, user_id=user_id, rating=align_rating(rating_value)
        )
    except Exception as e:
        print("Error adding rating to database: " + str(e))
        return False

    # extra step: we have to generate the tags for the image
    # but that will be the API's job
    # we just return True here

    return True


def remove_rating(filename, discord_id):
    # this simply removes all ratings for a given image and user
    # first we need to get the image.id
    image_id = Image.select(Image.id).where(Image.filename == filename).dicts()

    if len(image_id) == 0:
        return False
    else:
        image_id = image_id[0]["id"]

    # now we have to get the user.id
    user_id = User.select(User.id).where(User.discord_id == discord_id).dicts()

    if len(user_id) == 0:
        return False
    else:
        user_id = user_id[0]["id"]

    # now we can delete the rating
    (
        Rating.delete()
        .where(Rating.image_id == image_id, Rating.user_id == user_id)
        .execute()
    )

    return True


def delete_image(filename):
    image_id = Image.select(Image.id).where(Image.filename == filename).dicts()

    if len(image_id) == 0:
        return False
    else:
        image_id = image_id[0]["id"]

    (Rating.delete().where(Rating.image_id == image_id).execute())

    (ImageTag.delete().where(ImageTag.image_id == image_id).execute())

    (Image.delete().where(Image.id == image_id).execute())

    return True


def get_dataset_stats(discord_id):
    data = get_userdata(discord_id)

    summary = [0, 0, 0, 0, 0, 0, 0]
    for i in range(len(data)):
        rating = round(data[i]["rating"] * 6)
        summary[int(rating)] += 1

    from modules.utils import checkpoint_dataset_hash, raternn_up_to_date

    raternnp_hash = checkpoint_dataset_hash(
        "models/RaterNNP_" + str(discord_id) + ".pth"
    )
    dataset_hash = create_RPDataset(discord_id).generate_hash()

    stats = {
        "image_count": len(data),
        "rating_distribution": summary,
        "RaterNNP_up_to_date": raternnp_hash == dataset_hash,
        "RaterNN_up_to_date": raternn_up_to_date(get_discord_ids()),
    }
    return stats


# These following functions are for training the RaterNNP and RaterNN models
def create_RPDataset(discord_id):
    # first we want the ratings of the users, with the image_512_t images
    ratings = (
        Rating.select(Rating.rating, Image.image_512_t)
        .join(Image)
        .switch(Rating)
        .join(User)
        .where(User.discord_id == discord_id)
        .order_by(Image.id)
        .dicts()
    )

    formatted_ratings = []
    for rating in ratings:
        formatted_ratings.append(
            {
                "image": rating["image_512_t"],
                "rating": rating["rating"],
            }
        )

    # now we can create the dataset
    from modules.datasets import RPDataset
    from modules.utils import get_val_transforms

    dataset = RPDataset(
        data=formatted_ratings, username=discord_id, transform=get_val_transforms()
    )

    return dataset
