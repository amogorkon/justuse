# korean characters: https://github.com/arcsecw/wubi/blob/master/wubi/cw.py
# chinese characters: https://github.com/tsroten/zhon/blob/develop/zhon/cedict/all.py

from hash_alphabet_chinese import chinese_characters
from hash_alphabet_emojis import emojis
from hash_alphabet_korean import korean_characters

ascii_characters = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!#$%&()*+,-./:;<=>?@[]^_`{|}~"
alphabet = list(ascii_characters) + emojis + chinese_characters + korean_characters
alphabet = sorted(list(set(alphabet)))