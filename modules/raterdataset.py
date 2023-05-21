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

    # get image by filename
    def get_image(self, filename):
        # check if image is in the dataset
        if filename not in list(map(lambda x: x["image"], self.data)):
            return None
        with open(os.path.join(self.imagefolder, filename), "rb") as f:
            return f.read()