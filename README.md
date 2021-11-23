[![license](https://img.shields.io/github/license/amogorkon/justuse)](https://github.com/amogorkon/justuse/blob/main/LICENSE)
[![stars](https://img.shields.io/github/stars/amogorkon/justuse?style=plastic)](https://github.com/amogorkon/justuse/stargazers)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/amogorkon/justuse/graphs/commit-activity)
[![Updates](https://pyup.io/repos/github/amogorkon/justuse/shield.svg)](https://pyup.io/repos/github/amogorkon/justuse/)
[![Build](https://github.com/amogorkon/justuse/actions/workflows/blank.yml/badge.svg?branch=main)](https://github.com/amogorkon/justuse/actions/workflows/blank.yml)
[![Downloads](https://pepy.tech/badge/justuse)](https://pepy.tech/project/justuse)
[![justuse](https://snyk.io/advisor/python/justuse/badge.svg)](https://snyk.io/advisor/python/justuse)
[![slack](https://img.shields.io/badge/slack-@justuse-purple.svg?logo=slack)](https://join.slack.com/t/justuse/shared_invite/zt-tot4bhq9-_qIXBdeiRIfhoMjxu0EhFw)
[![codecov](https://codecov.io/gh/amogorkon/justuse/branch/unstable/graph/badge.svg?token=ROM5GP7YGV)](https://codecov.io/gh/amogorkon/justuse)
[![Sourcery](https://img.shields.io/badge/Sourcery-enabled-brightgreen)](https://sourcery.ai)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

[logo]: https://github.com/amogorkon/justuse/blob/unstable/logo.svg "Logo"

# Just use() python the way you want!

## Installation
To install, enter `python -m pip install justuse` in a commandline, then you can `import use` in your code and simply use() stuff. Check for examples below and our [Showcase](https://github.com/amogorkon/justuse/blob/unstable/docs/Showcase.ipynb)!

## Features, Claims & Goals

- [x] inline version-checking
- [x] safely import code from an online URL - towards an unhackable infrastructure ("Rather die than get corrupted!")
- [x] initial module globals - a straight forward solution to diamond/cyclic imports
- [x] decorate all specified callables (functions, methods, classes, ..) on import via pattern matching, aspect-orientation made easy
- [x] return a given default if an exception happened during an import, simplifying optional dependencies
- [x] safe hot auto-reloading of function-only local modules - a REPL-like dev experience with files in jupyter and regular python interpreters
- [x] safely auto-install version-tagged pure python packages from PyPI (packages with C-extensions like numpy don't work yet)
- [x] have multiple versions of the same package installed and loaded in the same program without conflicts
- [x] auto-install packages with C-extensions and other precompiled stuff
- [x] no-hassle inline auto-installation of (almost) all conda packages
- [ ] attach birdseye debugger to a loaded module as a mode
- [ ] try to pull packages from a P2P network before pulling from PyPI or conda directly
- [ ] all justuse-code is compiled to a single, standalone .py file - just drop it into your own code without installation
- [ ] provide a visual representation of the internal dependency graph
- [ ] module-level variable guards aka "module-properties"
- [ ] isolation of packages via subprocess/subinterpreter for clean un-/reloading
- [ ] slot-based plugin architecture (to ensure reproducable testability of plugin-combinations)
- [ ] document everything!
- [ ] test everything!

## The Story
Over the years, many times I've come across various situations where Python's import statement just didn't work the way I needed.
There were projects where I felt that a central module from where to expand functionality would be the simplest, most elegant approach, but that would only work with simple modules and libs, not with functionality that required access to the main state of the application. In those situations the first thing to try would be "import B" in module A and "import A" in module B - a classical circular import, which comes with a lot of headaches and often results in overly convoluted code. All this could be simplified if it was possible to pass some module-level global variables to the about-to-be-imported module before its actual execution, but how the heck could that work with an import statement?

Later, I worked and experimented a lot in jupyter notebooks. Usually I would have some regular python module only with pure functions open in an editor, import that module in jupyter and have all the state there. Most of the time I'd spend in jupyter, but from time to time I'd edit the file and have to manually reload the module in jupyter - why couldn't it just automatically reload the module when my file is saved? Also, in the past reload() was a builtin function, now I have to import it extra from importlib, another hoop to jump through..

Then there were situations where I found a cool module on github, a single file in a package I wanted to import - but why is it necessary to download it manually or maybe pip install an outdated package? Shouldn't it be possible to import just *this exact file* and somehow make it secure enough that I don't have to worry about man-in-the-middle attacks or some github-hack?

On my journey I came across many blogposts, papers and presentation-notebooks with code that just 'import library' but mentions nowhere which actual version is being used, so  trying to copy and paste this code in order to reproduce the presented experiment or solution is doomed to failure.

I also remember how I had some code in a jupyter notebook that did 'import opencv' but I had not noted which actual version I had initially used. When I tried to run this notebook after a year again, it failed in a subtle way: the call signature of an opencv-function had slightly changed. It took quite a while to track down what my code was expecting and when this change occured until I figured out how to fix this issue. This could've been avoided or at least made a lot simpler if my imports were somehow annotated and checked in a pythonic way. After I complained about this in IRC, nedbat suggested an alternative functional way for imports with assignments: `mod = import("name", version)` which I found very alien and cumbersome at first sight - after all, we have an import statement for imports, and *there should be one and preferrably only one way to do it* - right?

Well, those shortcomings of the import statement kept bugging me. And when I stumbled over the word 'use' as a name for whatever I was conceiving, I thought "what the heck, let's try it! - how hard could it be?!" Turns out, some challenges like actual, working hot-reloading are pretty hard! But by now use() can cover all the original usecases and there's even more to come!

# Examples
Here are a few tidbits on how to use() stuff to wet your appetite, for a more in-depth overview, check out our [Showcase](https://github.com/amogorkon/justuse/blob/unstable/docs/Showcase.ipynb)!

 `import use`
 
 `np = use("numpy", version="1.19.2")` corresponds to `import numpy as np; if np.version != "1.19.2": warn()`
 
 `use("pprint").pprint(some_dict)` corresponds to a one-off `from pprint import pprint; pprint(some_dict)` without importing it into the global namespace
 
 `tools = use(use.Path("/media/sf_Dropbox/code/tools.py"), reloading=True)` impossible to realize with classical imports
 
 `test = use("functions", initial_globals={"foo":34, "bar":49})` also impossible with the classical import statement, although importlib can help
 
 `utils = use(use.URL("https://raw.githubusercontent.com/PIA-Group/BioSPPy/7696d682dc3aafc898cd9161f946ea87db4fed7f/biosppy/utils.py"),
            hashes={"95f98f25ef8cfa0102642ea5babbe6dde3e3a19d411db9164af53a9b4cdcccd8"})` no chance with classical imports
            
 `np = use("numpy", version="1.21.0rc2", hashes={"3c90b0bb77615bda5e007cfa4c53eb6097ecc82e247726e0eb138fcda769b45d"}, modes=use.auto_install)` inline installation of packages and importing the same package with different versions in parallel in the same code - most people wouldn't even dream of that!

## There are strange chinese symbols in my hashes, am I being hacked?
Nope. SHA256 hashes normally are pretty long (64 characters per hexdigest) and we require them defined within regular python code. Additionally, if you want to support multiple platforms, you need to supply a hash for every platform - which can add up to huge blocks of visual noise. Since Python 3 supports unicode by default, why not use the whole range of printable characters to encode those hashes? It's easier said than done - turns out emojis don't work well across different systems and editors - however, it *is* feasible to merge the Japanese, ASCII, Chinese and Korean alphabets into a single, big one we call JACK - which can be used to reliably encode those hashes in merely 18 characters. Since humans aren't supposed to manually type those hashes but simply copy&paste them anyway, there is only the question how to get them if you only have hexdigests at hand for some reason. Simply do `use.hexdigest_as_JACK(H)` and you're ready to go. Of course we also support classical hexdigests as fallback.

## Beware of Magic!
Inspired by the q package/module, use() is a subclass of ModuleType, which is a callable class that replaces the module on import, so that only 'import use' is needed to be able to call use() on things.

Probably the most magical thing about use() is that it does not return the plain module you wanted but a *ProxyModule* instead which adds a layer of abstraction. This allows things like automatic and transparent reloading without any intervention needed on your part. ProxyModules also add operations on modules like aspectizing via `mod @ (check, pattern, decorator)` syntax, which would not be possible with the classical import machinery.
