import torch
from PIL import Image
import os

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

    def add_image(self, image, rating, found=False):
        if image.filename in list(map(lambda x: x["image"], self.data)):
            index = list(map(lambda x: x["image"], self.data)).index(image.filename)
            self.data[index]["rating"] = rating
            return self.data[index]
        else:
            self.data.append({"image": image.filename, "rating": rating})
            if not found:
                image.save(os.path.join(self.imagefolder, image.filename))
            return self.data[-1]
        
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
class RTData():
    def __init__(self, dataset_json, transform=None):
        self.transform = transform
        self.usersets, self.usernames, self.imagefolder, self.full_dataset = self.load_dataset(
            dataset_json
        )

    def load_dataset(self, dataset_json):
        import json

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
        for i in range(len(dataset["userdata"])):
            entry = RPDataset(
                dataset["userdata"][i], dataset["imagefolder"], self.transform
            )
            usersets.append(entry)

        if "full_dataset" in dataset:
            full_dataset = RDataset(
                dataset["full_dataset"], dataset["usernames"], dataset["imagefolder"], self.transform
            )
        else:
            full_dataset = []
        return usersets, dataset["usernames"], dataset["imagefolder"], full_dataset

    def get_userdataset(self, username):
        index = self.usernames.index(username)
        return self.usersets[index]

    def get_image(self, filename):
        # check if image exists on disk
        if os.path.isfile(os.path.join(self.imagefolder, filename)):
            with open(os.path.join(self.imagefolder, filename), "rb") as f:
                return f.read()
        else:
            return None

    def save_dataset(self, path):
        import json

        with open(path, "w") as f:
            json.dump(
                {
                    "userdata": list(map(lambda x: x.toList(), self.usersets)),
                    "usernames": self.usernames,
                    "imagefolder": self.imagefolder,
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

        item = self.usersets[user_index].add_image(image, rating, found)
        return item
    
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
                    index = list(map(lambda x: x["image"], userset)).index(data[i]["image"])
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
                batches.append(copy.deepcopy(userdata[j:j+batch_size])) # we have to copy because we need to break the reference

            # now we have to predict the ratings for each batch
            # we can use ratermodels[i].rateImageBatch(images) to predict the ratings in a batch
            loop = tqdm.tqdm(range(len(batches)))
            for j in loop:
                images = []
                for k in range(len(batches[j])):
                    images.append(os.path.join(self.imagefolder, batches[j][k]["image"]))
                ratings = ratermodels[i].rateImageBatch(images)
                for k in range(len(batches[j])):
                    batches[j][k]["ratings"][i] = ratings[k]
                loop.set_description(f"Predicting {self.usernames[i]}'s ratings for batch {j+1}/{len(batches)}")

            # now we have to add the ratings back to the data from the batches
            for j in range(len(batches)):
                for k in range(len(batches[j])):
                    index = list(map(lambda x: x["image"], data)).index(batches[j][k]["image"])
                    if data[index]["ratings"][i] == -1:
                        data[index]["ratings"][i] = batches[j][k]["ratings"][i]


        # now we have the full dataset
        self.full_dataset = RDataset(data, self.usernames, self.imagefolder, self.transform)
        return self.full_dataset
        
    # this function verifies if the full dataset's human ratings are correct (it is the same as the usersets)    
    def verify_full_dataset(self):
        # iterate through each userset
        for i in range(len(self.usersets)):
            # iterate through each item in the given userset
            for j in range(len(self.usersets[i].toList())):
                # find the item in the full dataset
                index = list(map(lambda x: x["image"], self.full_dataset.toList())).index(self.usersets[i].toList()[j]["image"])
                # check if the ratings are the same
                if self.usersets[i].toList()[j]["rating"] != self.full_dataset.toList()[index]["ratings"][i]:
                    # print out which image and user has a different rating
                    return False
        return True