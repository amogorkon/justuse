"""
A self-documenting, improved way to import modules in Python.

Goals:
- version check on the spot, potential version conflicts become obvious (done)
- load standalone-modules from online sources with hash-check (done)
- auto-reload if content is changed, checked in given interval or on file-change (WIP)
- pass module-level globals into the importing context (TODO)
- easy introspection of internal dependency graph (TODO)
- relative imports on online-sources (TODO)

Examples:
>>> from use import use

# equivalent to `import numpy as np` with explicit version check
>>> np = use("numpy", version="1.1.1")
>>> np.version == "1.1.1"
True

# equivalent to `from pprint import pprint; pprint(dictionary)` but without assigning 
# pprint to a global variable, thus no namespace pollution
>>> use("pprint").pprint([1,2,3])
[1,2,3]
# equivalent to sys.path manipulation, then `import tools` with a reload(tools) every 2 secs if the file changed
>>> tools = use("/media/sf_Dropbox/code/tools.py", reloading=2)

# it is possible to import standalone modules (which only import stdlib or use() other online-sources)
# with immediate sha1-hash-verificiation before execution of the code
>>> test = use(URL("https://raw.githubusercontent.com/PIA-Group/BioSPPy/7696d682dc3aafc898cd9161f946ea87db4fed7f/biosppy/utils.py"),
                    hash_value="77fe711647cd80e6a01668dba7f2b9feb8f435ba")

File-Hashing inspired by 
- https://github.com/kalafut/py-imohash
- https://github.com/fmoo/python-varint/blob/master/varint.py

:author: github@anselm.kiefner.de (Anselm Kiefner)
:license: MIT
"""


import sys, os

import imp

import hashlib

import inspect

import asyncio
import mmh3

from collections import namedtuple
from importlib import import_module
from importlib import reload
from packaging.version import parse
from pathlib import Path
from types import ModuleType
from warnings import warn

import mmh3
import requests

from yarl import URL

__version__ = "0.0.1"

class VersionWarning(Warning):
    pass

class NotReloadableWarning(Warning):
    pass

class ModuleNotFoundError(RuntimeWarning):
    pass

class NoValidationWarning(Warning):
    pass

class UnexpectedHash(RuntimeWarning):
    pass

Using = namedtuple("UsedModule", "source version hash reloading frame")

def varint_encode(number):
    """Pack `number` into varint bytes"""
    buf = b''
    while True:
        towrite = number & 0x7f
        number >>= 7
        if number:
            buf += bytes((towrite | 0x80,))
        else:
            buf += bytes((towrite,))
            break
    return buf

def hashfileobject(filename, sample_threshhold=128 * 1024, sample_size=16 * 1024, hexdigest=False):
    with open(filename, 'rb') as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(0, os.SEEK_SET)

        if size < sample_threshhold or sample_size < 1:
            data = f.read()
        else:
            data = f.read(sample_size)
            f.seek(size//2)
            data += f.read(sample_size)
            f.seek(-sample_size, os.SEEK_END)
            data += f.read(sample_size)

        hash_tmp = mmh3.hash_bytes(data)
        hash_ = hash_tmp[7::-1] + hash_tmp[16:7:-1]
        enc_size = varint_encode(size)
        digest = enc_size + hash_[len(enc_size):]
        return digest


__using__ = {}
            

class SurrogateModule(ModuleType):
    def __init__(self, mod, interval=1):
        self.__implementation = mod
        
        async def __reload():
            while True:
                await asyncio.sleep(interval)
                try:
                    mod = reload(self.__implementation.__name__)
                    self.__implementation = mod
                except Exception as e:
                    print(e)
                
        loop = asyncio.get_event_loop()
        loop.create_task(__reload())

    def __getattr__(self, name):
        return getattr(self.__implementation, name)
    
    def __setattr__(self, name, value):
        if name == "_SurrogateModule__implementation":
            object.__setattr__(self, name, value)
        setattr(self.__implementation, name, value)

def use(thing, version:str=None, reloading:int=0, hash_algo="sha1", hash_value=None):
    if isinstance(thing, Path):
        path_set = set(sys.path)
        if thing.is_dir():
            raise RuntimeWarning(f"Can't import directory {thing}")
        elif thing.is_file():
            path_set.add(str(thing.resolve(strict=True).parents[0]))
            sys.path = list(path_set)
            name = thing.stem
            mod = import_module(name)
        else:
            raise ModuleNotFoundError(f"Are you sure '{thing}' exists?")
    elif isinstance(thing, URL):
        response = requests.get(thing)
        if not response.status_code == 200:
            raise ModuleNotFoundError(f"Could not load {thing} from the interwebs, got a {response.status_code} error.")
        name = thing.name
        if hash_value:
            if hash_algo == "sha1":
                if not (this_hash := hashlib.sha1(response.content).hexdigest()) == hash_value:
                    raise UnexpectedHash(f"{this_hash} does not match the expected hash {hash_value} - aborting!")
            else:
                raise NotImplementedError
        else:
            warn("Attempting to import {thing} from the interwebs with no validation whatsoever! This means we are flying blind and are possibly prone to man-in-the-middle attacks.")
        mod = imp.new_module(name)
        exec(compile(response.content, name, "exec"), mod.__dict__)

    else:
        name = thing
        mod = import_module(name)
    try:
        #source version hash reloading frame
        source = mod.__file__
    except AttributeError:
        source = thing
    __using__[name] = Using(source, version, hash_value, reloading, inspect.getframeinfo(inspect.currentframe()))
    if version:
        try:
            if parse(str(version)) != parse(str(mod.__version__)):
                warn(
                    f"{name} is expected to be version {version} ,  but got {mod.__version__} instead",
                    VersionWarning,
                )
        except AttributeError:
            print(f"Cannot determine version for module {name}, continueing.")

    if reloading:
        if not getattr(mod, "__file__", False):
            warn(
                f"{name} does not look like a real file, reloading is not possible.",
                NotReloadableWarning,
            )
        if not getattr(mod, "__reloadable__", False):
            warn(
                f"Beware {name} is not flagged as reloadable, things may easily break!",
                NotReloadableWarning,
            )
        mod = SurrogateModule(mod)
        __using__[name] = mod


    return mod
