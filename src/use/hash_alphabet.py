# korean characters: https://github.com/arcsecw/wubi/blob/master/wubi/cw.py
# chinese characters: https://github.com/tsroten/zhon/blob/develop/zhon/cedict/all.py

from hash_alphabet_chinese import chinese_characters
from hash_alphabet_emojis import emojis
from hash_alphabet_japanese import japanese_characters
from hash_alphabet_korean import korean_characters

ascii_characters = (
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!#$%&()*+,-./:;<=>?@[]^_`{|}~"
)
# emojis only cause interoperability issues
alphabet = list(ascii_characters) + chinese_characters + korean_characters
alphabet = sorted(list(set(alphabet)))
reverse_alphabet = {c: i for i, c in enumerate(alphabet)}


def represent_num_as_base(num, base):
    if num == 0:
        return [0]
    digits = []
    while num:
        digits.append(num % base)
        num //= base
    return digits[::-1]


def hexdigest_as_CJK(string):
    if not string:
        return
    return "".join(alphabet[c] for c in represent_num_as_base(int(string, 16), len(alphabet)))


def CJK_as_num(string):
    return sum(len(reverse_alphabet) ** i * reverse_alphabet[x] for i, x in enumerate(reversed(string)))


def num_as_hexdigest(num):
    return hex(num)[2:]
