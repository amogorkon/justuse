import logging
import os, re, shlex, sys
from typing import Iterator

logging.root.setLevel(logging.DEBUG)
logger = logging.getLogger("air")
def log(msg, *args, **kwargs):
  text = msg
  try:
    text = msg.format(*args, **kwargs)
  except:
    try:
      text = msg % args
    except:
      text = ", ".join([msg, *args, *kwargs.items()])
  logger.log(logging.WARN, text)

# In[1]:
def install_packages(*packages):
   extra_options = []
   if "EXTRA_PIP_OPTIONS" in os.environ:
     extra_options = shlex.split(os.environ["EXTRA_PIP_OPTIONS"])
     log("Adding extra pip options: {}", extra_options)
   
   log("Installing packages: {}", packages)
   import sys
   __oldexit = sys.exit
   try:
     sys.exit = lambda *args: exec(
     "raise BaseException(\x27\x27.join(map(str,args)))")
     import pip._internal.commands.install
     c = pip._internal.commands.install.InstallCommand("install", ""); 
     ctx: Iterator[None] = c.main_context()
     runner = ctx(lambda: c._main([
       "--progress-bar", "off", "--no-input", "--pre",
       "--no-build-isolation", "--ignore-requires-python",
       "--prefer-binary", *extra_options, *packages
     ]))
     return runner()
   finally:
     sys.exit = __oldexit

log("Importing modules")
try:
  import beartype
except ImportError:
  install_packages("beartype")

from use import use
