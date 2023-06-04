class Config(object):
    import json

    configfile = "config.json"

    with open(configfile) as json_data_file:
        data = json.load(json_data_file)
    
    db_host = data["db_host"]
    db_port = data["db_port"]
    db_name = data["db_name"]
    db_user = data["db_user"]
    db_password = data["db_password"]

    rater = data["rater"]
    T11 = data["T11"]