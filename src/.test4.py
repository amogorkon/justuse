hashes = {("a", "aa", "aaa"), ("b", "bb", "bbb")}
name = "pygame"
version = "1.2.3"



hash_str = "{\n    " + "\n    ".join(f'{H} # {python}-{platform}' for python, platform, H in hashes) + " \n}"
s =  f"use('{document["name"]}', version='{version}', modes=use.auto_install, hash_algo=use.Hash.sha256, hashes={hash_str})"
print(s)