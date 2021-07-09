import toml

config = {"1":2, "2":3}

with open("config.json", "w") as file:
    toml.dump(config, file)