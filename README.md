# Main API Endpoints

## Default Endpoints

| Endpoint      | Method | Purpose                      | Params                              |
| ------------- | ------ | ---------------------------- | ----------------------------------- |
| /getimage     | GET    | Returns an image file        | args: {"image":"imagefilename.jpg"} |
| /getimagetags | GET    | Returns a given image's tags | args: {"image":"imagefilename.jpg"} | 

## TaggerNN Endpoints

| Endpoint | Method | Purpose             | Params                                    | 
| -------- | ------ | ------------------- | ----------------------------------------- | 
| /tag     | POST   | Tag an image        | form-data: {"image" (file): imagefile}    |
| /tagbulk | POST   | Tag multiple images | form-data: {"images" (files): imagefiles} |

## RaterNN Endpoints

| Endpoint  | Method | Purpose              | Params                                                                       |
| --------- | ------ | -------------------- | ---------------------------------------------------------------------------- |
| /rate     | POST   | Rate an image        | form-data: <br>{"image" (file): imagefile, <br> "user": "username or all"}   |
| /ratebulk | POST   | Rate multiple images | form-data: <br>{"images" (files): imagefiles,<br> "user": "username or all"} |

## Dataset Management Endpoints

| Endpoint        | Method | Purpose                                                                                   | Params                                                               |
| --------------- | ------ | ----------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| /getuserdata    | GET    | Gives back the user's dataset                                                             | args:<br>{"user":"username"}                                         |
| /getuserdata    | GET    | Same but filtered with tags                                                               | args:<br>{"user":"username",<br>"filters": "tag1,tag2,..."}          |
| /verifydatasets | GET    | Checks if the full dataset is valid<br>(there are new userratings since the last creation) | -                                                                    |
| /updatetags     | GET    | Creates tags for the new images without generated tags                                    | -                                                                    |
| /addrating      | POST   | Adds user rating to their dataset                                                         | form-data: {"image": imagefile,<br>"user":"username","rating":"0.5"} |

## Training Endpoints

| Endpoint       | Method | Purpose                                         | Params                    |
| -------------- | ------ | ----------------------------------------------- | ------------------------- |
| /trainerstatus | GET    | Returns the trainer's status                    | -                         |
| /trainuser     | GET    | Starts training the given user's RaterNNP model | args: {"user":"username"} |
| /stoptraining  | GET    | Stops the training if any is running            | -                         | 
