import use

use('thefuzz', version='0.19', modes=use.auto_install, hash_algo=use.Hash.sha256, hashes={
    'N頓盥㳍藿鞡傫韨司䧷焲瘴䵉紅蚥鲅獳陷',  # py2.py3-any 
})
from thefuzz import fuzz
from thefuzz import process