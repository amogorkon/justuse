

import importlib, itertools, functools, re, requests, sys
from importlib.machinery import EXTENSION_SUFFIXES
class ArtifactMatcher(object):
    def __init__(self, releases):
        self.rels = releases
    
    def arch(*args):
        suffixes = EXTENSION_SUFFIXES[0]
        pts = re.split(
          r"((?:py|cpy|pypy|cpython)-?[23]\.?(?:1?[0-9])|linux|\.dll|dylib|pyd|so|gnu|freebsd|manylinux1|win)(?=$|[^0-9])-?",
          suffixes)
        
        match_pts = [*filter(None, map(lambda i: i.strip("-") if not re.search(r"^(c|py|^)py(thon|)|^[23]\.?1?[0-9]",i) else None , functools.reduce(list.__add__, [*filter(lambda i: len(i) and i[0]!="." and i[0] and i[0] not in ("so","pyd","dll","dylib","gnu"), (re.split(r"(?<=[^0-9])(?=[23]\.?(?:1?[0-9]))",p) for p in  pts))],[])))]
        return match_pts
    
    def parse_version(self, info):
        py_ver = info["python_version"]
        py_ver_num = re.subn("[^0-9]+", "", py_ver)[0]
        if not py_ver_num:
          return (0, 0)
        py_major = int(py_ver_num[0],10)
        py_minor = int(py_ver_num[1],10)
        return (py_major, py_minor)
    
    def matches_sys_version(self, parsed_ver):
        sys_ver = tuple(sys.version_info[0:2])
        if sys_ver[0] == 3 and sys_ver[1] > 9 and \
           parsed_ver[0] == 3 and parsed_ver[1] == 9:
            return True
        return sys_ver == parsed_ver
    
    def score(self, info):
        arch_pts = self.arch()
        fn = info["filename"]
        fn2 = re.subn(
          "[1-9][0-9]*(?:\.[0-9]+)?", "-", info["filename"])[0]
        fnpts = re.split("[^a-zA-Z0-9]+", fn + fn2)
        fnpts += re.split("[^a-zA-Z0-9_]+", fn + fn2)
        score = 0
        for arch_pt in arch_pts:
          if arch_pt in fnpts:
            score += 1
        parsed_ver = self.parse_version(info)
        if self.matches_sys_version(parsed_ver):
          score += 1
        return score
    
    def __iter__(self):
        for ver, infos in self.rels.items():
            for info in infos:
                info["version"] = ver
                yield info;
    
    def get_sample_data(*args):
        return  requests.get(
        "https://raw.githubusercontent.com/greyblue9/junk/master/rels.json"
        ).json()
    
    def best(self):
        return [*sorted(self, key=self.score)][-1]
