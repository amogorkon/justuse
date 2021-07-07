import json


config = {}
with open("config.json") as file:
    try:
        config.update(json.load(file))
    except json.decoder.JSONDecodeError:
        pass