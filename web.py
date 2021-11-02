from flask import Flask, jsonify, request
from flask_cors import CORS
import math
import json
import grequests

app = Flask("aibackend")
CORS(app)

# load users.json
with open("models/users.json") as f:
    users = json.load(f)

# Worker Class
class Worker:
    def __init__(self, ip, port, name, quality):
        self.id = len(world.workers)
        self.ip = ip
        self.port = port
        self.name = name
        self.quality = quality

    def tag(self, image):
        url = f"http://{self.ip}:{self.port}/tag"
        files = {"image": image}
        r = grequests.post(url, files=files)
        return r.json()

    def rate(self, user, image):
        url = f"http://{self.ip}:{self.port}/rate"
        files = {"image": image}
        data = {"user": user}
        r = grequests.post(url, files=files, data=data)
        return r.json()


class World:
    def __init__(self):
        self.quality = 0
        self.workers = []

    def addWorker(self, ip, port, name, quality):
        worker = Worker(ip, port, name, quality)
        self.quality += quality
        self.workers.append(worker)
        return worker

    def workerCount(self):
        return len(self.workers)

    def getFastestWorker(self):
        fastestWorkerId = 0
        for (i, worker) in enumerate(self.workers):
            if worker.quality > self.workers[fastestWorkerId].quality:
                fastestWorkerId = i
        return self.workers[fastestWorkerId]


world = World()


def DistributeRateBulk(user, images):
    imagearrays = []
    imageslen = len(images)
    for i in range(0, world.workerCount()):
        imagearrays.append([])
        taskcount = math.trunc(imageslen / world.quality * world.workers[i].quality)
        for j in range(0, taskcount):
            imagearrays[i].append(images[0])
            images.pop(0)
    if len(images) > 0:
        imagearrays[world.getFastestWorker().id].extend(images)

    reqs = []
    for i in range(0, world.workerCount()):
        url = f"http://{world.workers[i].ip}:{world.workers[i].port}/ratebulk"
        files = []
        for image in imagearrays[i]:
            files.append(("images", image))
        data = {"user": user}
        req = grequests.post(url, files=files, data=data)
        reqs.append(req)
    print(f"Sending {len(reqs)} requests")
    results = grequests.map(reqs)
    print(f"Tasks completed")
    if user == "all":
        output = {"ratings": [], "users": users}
    else:
        output = {"ratings": []}
    for i in range(0, world.workerCount()):
        output["ratings"].extend(results[i].json()["ratings"])
    return output


@app.route("/registerworker", methods=["POST"])
def registerworker():
    worker = request.get_json()
    name = worker["name"]
    quality = worker["quality"]
    ip = worker["ip"]
    port = worker["port"]
    if (ip, port) not in world.workers:
        w = world.addWorker(ip, port, name, quality)
        print(f"Registered worker id:{w.id} name:{w.name}")
        return jsonify({"id": w.id, "name": w.name, "quality": w.quality})
    else:
        return jsonify({"error": "Worker already registered"})


@app.route("/rate", methods=["POST"])
def rate():
    user = request.form["user"]
    image = request.files["image"]
    worker = world.getFastestWorker()
    if user == "all":
        return jsonify(worker.rate(user, image))
    if user in users:
        return jsonify(worker.rate(user, image))
    else:
        return jsonify({"error": "User not found"})


@app.route("/ratebulk", methods=["POST"])
def ratebulk():
    images = request.files.getlist("images")
    user = request.form.get("user")
    if user == "all":
        r = DistributeRateBulk(user, images)
        return jsonify(r)
    else:
        if user in users:
            r = DistributeRateBulk(user, images)
            return jsonify(r)
        else:
            return jsonify({"error": "User not found"})


if __name__ == "__main__":
    from waitress import serve

    serve(app, host="0.0.0.0", port=2444)
    print("Server started")
