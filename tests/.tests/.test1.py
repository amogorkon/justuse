from importlib import metadata, util
import configparser
from pathlib import Path

home = Path.home() / "justuse-python"

config = configparser.ConfigParser()

with open(home/"config.ini") as file:
    config.read(file)

D = list(metadata.distributions())

spec = util.find_spec("math")

d = D[0]
