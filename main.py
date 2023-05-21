import json
import torch
from modules.models import *
from modules.utils import *
import os


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def tagImage(image_path):
    image = load_image(image_path)
    image = image.to(device)
    # T11 is the tagger model
    output = Tagger(image)
    # convert output to a list of tags and their probabilities, sorted by probability
    output = list(zip(Tagger.classes, output.tolist()[0]))
    # sort
    output.sort(key=lambda x: x[1], reverse=True)
    # only contain tags with probability > 0.5, but keep the probabilities
    output = list(filter(lambda x: x[1] > 0.5, output))
    return output


def rateImage(image_path):
    image = load_image(image_path)
    image = image.to(device)
    # rater is the rater model
    output = Rater(image)
    # convert output to a list of ratings and their probabilities, sorted by probability
    output = list(zip(Rater.usernames, output.tolist()[0]))
    # sort
    output.sort(key=lambda x: x[1], reverse=True)
    return output


def rateImageBatch(image_paths):
    images = [load_image(image_path, unsqueeze=False) for image_path in image_paths]
    images = torch.stack(images)
    images = images.to(device)
    # rater is the rater model
    output = Rater(images)
    ratings = []
    # add usernames to output
    for i in range(len(output)):
        ratings.append(list(zip(Rater.usernames, output[i].tolist())))
    # sort
    for i in range(len(ratings)):
        ratings[i].sort(key=lambda x: x[1], reverse=True)
    ratings = list(zip(image_paths, ratings))

    return ratings


def main():
    global config
    config = load_configs()
    global Tagger, Rater
    Tagger, Rater = load_models(config, device=device)

    imagepaths = [
        "test/sample-258c8d6b5a476ddd7b55720b6c7078ea.jpg",
    ]
    import time

    start = time.time()
    print("start")
    ratings = rateImageBatch(imagepaths)
    print(f"{time.time() - start} s")
    with open("output.json", "w") as f:
        json.dump({"ratings": ratings}, f, indent=4)


if __name__ == "__main__":
    main()
