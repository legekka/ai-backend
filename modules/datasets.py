import torch
from PIL import Image
import os

from modules.models import EfficientNetV2S

# single user's dataset
class RPDataset(torch.utils.data.Dataset):
    def __init__(self, data, username, transform=None):
        self.data = data
        self.username = username
        self.transform = transform
        self.dataset_hash = self.generate_hash()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        import base64
        import io

        image = base64.b64decode(self.data[idx]["image"])
        fileobject = io.BytesIO(image)

        image = Image.open(fileobject).convert("RGB")
        if self.transform:
            image = self.transform(image)
            
        rating = torch.tensor(float(self.data[idx]["rating"]))
        rating = rating.unsqueeze(0)
        return image, rating

    def generate_hash(self):
        import hashlib

        imageandratings = list(
            map(lambda x: (x["image"], float(x["rating"])), self.data)
        )
        import json

        return hashlib.sha256(json.dumps(imageandratings).encode()).hexdigest()


# this is the full dataset for RaterNN based on the RPDatasets
class RDataset(torch.utils.data.Dataset):
    def __init__(self, data, usernames, imagefolder, transform=None):
        self.data = data
        self.usernames = usernames
        self.imagefolder = imagefolder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(
            os.path.join(self.imagefolder, self.data[idx]["image"])
        ).convert("RGB")
        if self.transform:
            image = self.transform(image)
        ratings = torch.tensor(self.data[idx]["ratings"])
        return image, ratings

    def toList(self):
        return self.data


# This class manages the data for RaterNNP personalized models, and also the full dataset for RaterNN
class RTData:
    def __init__(self, dataset_json, transform=None):
        self.transform = transform
        (
            self.usersets,
            self.usernames,
            self.imagefolder,
            self.imagefolder2x,
            self.tags,
            self.full_dataset,
            self.update_needed,
        ) = self.load_dataset(dataset_json)
        self.dataset_hash = self.generate_hash()

    def load_dataset(self, dataset_json):
        import json

        print("Loading dataset...", end="", flush=True)

        with open(dataset_json) as f:
            dataset = json.load(f)

        # convert each dicts to objects
        for i in range(len(dataset["userdata"])):
            for j in range(len(dataset["userdata"][i])):
                dataset["userdata"][i][j] = {
                    "image": dataset["userdata"][i][j]["image"],
                    "rating": dataset["userdata"][i][j]["rating"],
                }

        usersets = []
        update_needed = False
        for i in range(len(dataset["userdata"])):
            entry = RPDataset(
                data=dataset["userdata"][i],
                imagefolder=dataset["imagefolder"],
                username=dataset["usernames"][i],
                transform=self.transform,
            )
            if dataset["tags"]:
                for j in range(len(dataset["userdata"][i])):
                    # find the image in tags
                    index = 0
                    while (index < len(dataset["tags"])) and (
                        dataset["tags"][index]["image"]
                        != dataset["userdata"][i][j]["image"]
                    ):
                        index += 1
                    if index < len(dataset["tags"]):
                        entry.data[j]["tags"] = dataset["tags"][index]["tags"]
                    else:
                        print(
                            "!!! Image not found in tags: "
                            + dataset["userdata"][i][j]["image"]
                            + ", update needed !!!"
                        )
                        update_needed = True

            usersets.append(entry)

        if "full_dataset" in dataset:
            full_dataset = RDataset(
                dataset["full_dataset"],
                dataset["usernames"],
                dataset["imagefolder"],
                self.transform,
            )
        else:
            full_dataset = []
        print("Done!")
        return (
            usersets,
            dataset["usernames"],
            dataset["imagefolder"],
            dataset["imagefolder2x"],
            dataset["tags"],
            full_dataset,
            update_needed,
        )

    def generate_hash(self):
        import hashlib

        imageandratings = list(
            map(lambda x: (x["image"], x["ratings"]), self.full_dataset.toList())
        )
        import json

        return hashlib.sha256(json.dumps(imageandratings).encode()).hexdigest()

    def get_userdataset(self, username):
        index = self.usernames.index(username)
        return self.usersets[index]

    # this function returns the user's dataset filtered by the tags
    def get_userdataset_filtered(self, username, filter_tags):
        import copy

        index = self.usernames.index(username)
        userdataset = copy.deepcopy(self.usersets[index].toList())
        userdataset_filtered = list(
            filter(
                lambda x: all(elem in x["tags"] for elem in filter_tags), userdataset
            )
        )
        return userdataset_filtered

    def get_image(self, filename):
        # check if image exists on disk
        if os.path.isfile(os.path.join(self.imagefolder, filename)):
            with open(os.path.join(self.imagefolder, filename), "rb") as f:
                return f.read()
        else:
            return None

    def get_image_2x(self, filename):
        # check if image exists on disk
        if os.path.isfile(os.path.join(self.imagefolder2x, filename)):
            with open(os.path.join(self.imagefolder2x, filename), "rb") as f:
                return f.read()
        else:
            return None

    def get_image_tags(self, filename):
        if filename not in list(map(lambda x: x["image"], self.tags)):
            return None
        index = 0
        while index < len(self.tags) and self.tags[index]["image"] != filename:
            index += 1
        if index == len(self.tags):
            return None
        return self.tags[index]["tags"]

    def save_dataset(self, path):
        import json

        with open(path, "w") as f:
            json.dump(
                {
                    "usernames": self.usernames,
                    "imagefolder": self.imagefolder,
                    "imagefolder2x": self.imagefolder2x,
                    "tags": self.tags,
                    "userdata": list(
                        map(
                            lambda x: list(
                                map(
                                    lambda y: {
                                        "image": y["image"],
                                        "rating": y["rating"],
                                    },
                                    x.toList(),
                                )
                            ),
                            self.usersets,
                        )
                    ),
                    "full_dataset": self.full_dataset.toList(),
                },
                f,
                indent=4,
            )

    def add_rating(self, image, username, rating):
        user_index = self.usernames.index(username)

        # first we have to check if the image is already in the full dataset
        found = False
        for i in range(len(self.usersets)):
            userset = self.usersets[i].toList()
            if image.filename in list(map(lambda x: x["image"], userset)):
                found = True
                break

        item = self.usersets[user_index].add_image_rating(image, rating, found)
        if not found:
            from modules.utils import convert_to_image_768

            image768 = convert_to_image_768(image)
            image768.save(os.path.join(self.imagefolder2x, image.filename))
        return item

    def update_rating(self, filename, username, rating):
        user_index = self.usernames.index(username)
        return self.usersets[user_index].update_rating(filename, rating)

    def pre_verify_usersets_for_full_dataset(self):
        retraining_needed = []
        for i in range(len(self.usersets)):
            username = self.usernames[i]
            if self.usersets[i].get_stats(username)["RaterNNP_up_to_date"] == False:
                retraining_needed.append(username)
        if len(retraining_needed) > 0:
            return retraining_needed
        else:
            return None

    def create_full_dataset(self):
        # first we create the full image list
        data = []
        for i in range(len(self.usersets)):
            userset = self.usersets[i].toList()
            # add images that are not in the images list
            for j in range(len(userset)):
                if userset[j]["image"] not in list(map(lambda x: x["image"], data)):
                    data.append({"image": userset[j]["image"]})

        # now we have to add ratings for each user for each image
        for i in range(len(data)):
            data[i]["ratings"] = [-1] * len(self.usersets)

            for j in range(len(self.usersets)):
                userset = self.usersets[j].toList()
                # find the image in the user's dataset
                if data[i]["image"] in list(map(lambda x: x["image"], userset)):
                    index = list(map(lambda x: x["image"], userset)).index(
                        data[i]["image"]
                    )
                    data[i]["ratings"][j] = userset[index]["rating"]

        import tqdm
        import copy

        from modules.utils import load_personalized_models

        ratermodels = load_personalized_models()

        for i in range(len(self.usernames)):
            # first we filter out the data that has anything but -1 in the ratings
            userdata = []
            for j in range(len(data)):
                if data[j]["ratings"][i] == -1:
                    userdata.append(data[j])

            # we have to create batches of images from the userdata
            batch_size = 64
            batches = []
            for j in range(0, len(userdata), batch_size):
                batches.append(
                    copy.deepcopy(userdata[j : j + batch_size])
                )  # we have to copy because we need to break the reference

            # now we have to predict the ratings for each batch
            # we can use ratermodels[i].rateImageBatch(images) to predict the ratings in a batch
            loop = tqdm.tqdm(range(len(batches)))
            for j in loop:
                images = []
                for k in range(len(batches[j])):
                    images.append(
                        os.path.join(self.imagefolder, batches[j][k]["image"])
                    )
                ratings = ratermodels[i].rateImageBatch(images)
                for k in range(len(batches[j])):
                    batches[j][k]["ratings"][i] = ratings[k]
                loop.set_description(
                    f"Predicting {self.usernames[i]}'s ratings for batch {j+1}/{len(batches)}"
                )

            # now we have to add the ratings back to the data from the batches
            for j in range(len(batches)):
                for k in range(len(batches[j])):
                    index = list(map(lambda x: x["image"], data)).index(
                        batches[j][k]["image"]
                    )
                    if data[index]["ratings"][i] == -1:
                        data[index]["ratings"][i] = batches[j][k]["ratings"][i]

        # now we have the full dataset
        self.full_dataset = RDataset(
            data, self.usernames, self.imagefolder, self.transform
        )

        self.dataset_hash = self.full_dataset.get_hash()

        return self.full_dataset

    # this function verifies if the full dataset's human ratings are correct (it is the same as the usersets)
    def verify_full_dataset(self):
        # iterate through each userset
        for i in range(len(self.usersets)):
            # iterate through each item in the given userset
            for j in range(len(self.usersets[i].toList())):
                # find the item in the full dataset
                index = list(
                    map(lambda x: x["image"], self.full_dataset.toList())
                ).index(self.usersets[i].toList()[j]["image"])
                # check if the ratings are the same
                if (
                    self.usersets[i].toList()[j]["rating"]
                    != self.full_dataset.toList()[index]["ratings"][i]
                ):
                    # print out which image and user has a different rating
                    print(
                        f"Image {self.usersets[i].toList()[j]['image']} has a different rating for user {self.usernames[i]}"
                    )
                    return False
        return True

    # this function creates/updates the self.tags list, which contains tags for each image

    def update_tags(self, tagger: EfficientNetV2S):
        # first we have to check if each image is in the tags list
        for i in range(len(self.usersets)):
            userset = self.usersets[i].toList()
            for j in range(len(userset)):
                if userset[j]["image"] not in list(
                    map(lambda x: x["image"], self.tags)
                ):
                    self.tags.append({"image": userset[j]["image"], "tags": []})

        # now we will use tagger to predict tags for each image
        # create batches of images
        batch_size = 32
        batches = []
        for i in range(0, len(self.tags), batch_size):
            batches.append(self.tags[i : i + batch_size])

        import tqdm

        # now we have to predict the tags for each batch
        # we can use tagger.tagImageBatch(images) to predict the tags in a batch
        loop = tqdm.tqdm(range(len(batches)))
        for i in loop:
            images = []
            for j in range(len(batches[i])):
                images.append(os.path.join(self.imagefolder, batches[i][j]["image"]))
            tags = tagger.tagImageBatch(images)
            # now the tags have probabilities too, in the following structure:
            # [ [[tag1, prob1], [tag2, prob2], ...], [[tag1, prob1], [tag2, prob2], ...], ... ]
            # we only need the tags, so we will remove the probabilities
            for j in range(len(tags)):
                tags[j] = list(map(lambda x: x[0], tags[j]))
            # now we have to add the tags back to the data from the batches
            for j in range(len(batches[i])):
                batches[i][j]["tags"] = tags[j]
            loop.set_description(f"Predicting tags for batch {i+1}/{len(batches)}")

        # finally we have to add the tags back to the self.tags list
        for i in range(len(batches)):
            for j in range(len(batches[i])):
                index = list(map(lambda x: x["image"], self.tags)).index(
                    batches[i][j]["image"]
                )
                self.tags[index]["tags"] = batches[i][j]["tags"]

        # now we have the full tags list
        return self.tags

    def remove_duplicates(self):
        count = 0
        for userset in self.usersets:
            indexes_to_remove = []
            for i in range(len(userset.toList())):
                for j in range(i + 1, len(userset.toList())):
                    if userset.toList()[i]["image"] == userset.toList()[j]["image"]:
                        indexes_to_remove.append(j)
                        count += 1
            indexes_to_remove = sorted(indexes_to_remove, reverse=True)
            for index in indexes_to_remove:
                userset.toList().pop(index)
            userset.dataset_hash = userset.generate_hash()
        print(f"Total duplicates removed: {count}")
