import sys
import time
import logging
import use
from pathlib import Path

Observer = use('watchdog.observers', version='2.3.0', modes=use.auto_install, hash_algo=use.Hash.sha256, hashes={
    'Q鉨麲㺝翪峬夛冕廛䀳迆婃儈正㛣辐Ǵ娇',  # py3-win_amd64 
}).Observer

LoggingEventHandler = use('watchdog.events', version='2.3.0', modes=use.auto_install, hash_algo=use.Hash.sha256, hashes={
    'Q鉨麲㺝翪峬夛冕廛䀳迆婃儈正㛣辐Ǵ娇',  # py3-win_amd64 
}).LoggingEventHandler


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
path = Path(r"E:\Dropbox\code\_test")
event_handler = LoggingEventHandler()
observer = Observer()
observer.schedule(event_handler, path)
observer.start()
try:
    while True:
        time.sleep(1)
finally:
    observer.stop()
    observer.join()
