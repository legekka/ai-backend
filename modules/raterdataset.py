import torch
from PIL import Image
import os

from modules.models import EfficientNetV2S

# example data in dataset.json:
# dataset.data[0] = {
#    "image": "0001.jpg",
#    "ratings": [0, 0.2, ..., 1]
# }
# dataset.usernames contains the list of usernames


class RaterDataset(torch.utils.data.Dataset):
    # used for training the rater model
    def __init__(self, dataset_json, imagefolder, transform=None):
        self.data, self.usernames = self.load_dataset(dataset_json)
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

    # used for managing the dataset
    def load_dataset(self, path):
        import json

        with open(path) as f:
            dataset = json.load(f)
        return dataset["data"], dataset["usernames"]

    def save_dataset(self, path):
        import json

        with open(path, "w") as f:
            json.dump({"data": self.data, "usernames": self.usernames}, f, indent=4)

    def get_user_data(self, username):
        # returns each image and its ratings for the given user
        index = self.usernames.index(username)
        ratings = list(map(lambda x: x["ratings"][index], self.data))
        # add image names to each rating
        ratings = list(
            map(lambda x, y: ({"image": x["image"], "rating": y}), self.data, ratings)
        )
        return ratings

    # add/update rating (+ add image)
    def add_rating(self, image, user_and_rating, RaterNN):
        # user_and_rating is a simple object with like this:
        # {
        #    "username": "legekka",
        #    "rating": 0.5
        # }

        # user check should be done in api level,
        index = self.usernames.index(user_and_rating["username"])

        # we have to check if the image is already in the dataset
        # check if image.filename is in self.data[]["image"]
        if image.filename in list(map(lambda x: x["image"], self.data)):
            # image is already in the dataset we just update the rating
            image_index = list(map(lambda x: x["image"], self.data)).index(
                image.filename
            )
            self.data[image_index]["ratings"][index] = user_and_rating["rating"]
        else:
            # if image is not in the dataset, we will not just simply add it,
            # but generate ratings by RaterNN itself for all users,
            # then update the rating with the human data

            ratings = RaterNN.rateImage(image)

            ratings = list(map(lambda x: x[1], ratings))

            self.data.append({"image": image.filename, "ratings": ratings})

            image_index = len(self.data) - 1
            self.data[image_index]["ratings"][index] = user_and_rating["rating"]

            # save image to imagefolder
            # image.save(os.path.join(self.imagefolder, image.filename))

        return self.data[image_index]

    # get images by filenames
    def get_image(self, filename):
        # check if images is in the dataset
        if filename not in list(map(lambda x: x["image"], self.data)):
            return None
        else:
            with open(os.path.join(self.imagefolder, filename), "rb") as f:
                return f.read()


class RPDataset(torch.utils.data.Dataset):
    def __init__(self, data, imagefolder, transform=None):
        self.data = data
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
        rating = torch.tensor(float(self.data[idx]["rating"]))
        # dont forget that [batch] shape is deprecated, use [batch, 1]
        rating = rating.unsqueeze(0)
        return image, rating

    def toList(self):
        return self.data

    def add_image_rating(self, image, rating, found=False):
        if image.filename in list(map(lambda x: x["image"], self.data)):
            index = list(map(lambda x: x["image"], self.data)).index(image.filename)
            self.data[index]["rating"] = rating
            return self.data[index]
        else:
            self.data.append({"image": image.filename, "rating": rating})
            if not found:
                from modules.utils import process_image_for_dataset

                image512 = process_image_for_dataset(image)
                image512.save(os.path.join(self.imagefolder, image.filename))
            return self.data[-1]

    def update_rating(self, image, rating):
        i = 0
        while i < len(self.data) and self.data[i]["image"] != image:
            i += 1
        if i < len(self.data):
            self.data[i]["rating"] = rating
            return self.data[i]
        else:
            return None


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
            self.update_needed
        ) = self.load_dataset(dataset_json)
 
    
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
                dataset["userdata"][i], dataset["imagefolder"], self.transform
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
                        print("!!! Image not found in tags: " + dataset["userdata"][i][j]["image"] + ", update needed !!!")
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
            update_needed
        )

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
            from modules.utils import process_image_for_2x

            image768 = process_image_for_2x(image)
            image768.save(os.path.join(self.imagefolder2x, image.filename))
        return item

    def update_rating(self, filename, username, rating):
        user_index = self.usernames.index(username)
        return self.usersets[user_index].update_rating(filename, rating)

    def create_full_dataset(self, ratermodels):
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
