[![license](https://img.shields.io/github/license/amogorkon/justuse)](https://github.com/amogorkon/justuse/blob/master/LICENSE)
[![stars](https://img.shields.io/github/stars/amogorkon/justuse?style=plastic)](https://github.com/amogorkon/justuse/stargazers)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![Build](https://github.com/amogorkon/justuse/actions/workflows/blank.yml/badge.svg?branch=main)](https://github.com/amogorkon/justuse/actions/workflows/blank.yml)




# Just use() python modules the way you want!

## Installation
To install, enter `python -m pip install justuse` in a commandline, then you can `import use` in your code and simply use() stuff. Check for examples below!

## Features, Claims & Goals
Solid:
* inline version-checking
* safely import code from an online URL - towards an unhackable infrastructure ("Rather die than get corrupted!")
* initial module globals - a straight forward solution to diamond/cyclic imports
* decorate all specified callables (functions, modules, classes, ..) on import via pattern matching, aspect-orientation made easy
* return a given default if an exception happened during an import, simplifying optional dependencies

In flux:
* safely hot auto-reloading of function-only local modules - a REPL-like dev experience with files
* safely auto-install version-tagged packages from PyPI
* pull packages from a P2P network instead of PyPI directly
* provide a visual representation of the internal dependency graph
* test everything
* module-level variable guards aka "module-properties"

## The Story
Over the years many times I've come across various situations where Python's import statement just didn't work the way I needed.
There were projects where I felt that a central module from where to expand functionality would be the simplest, most elegant approach, but that would only work with simple modules and libs, not with functionality that required access to the main state of the application. In those situations the first thing to try would be "import B" in module A and "import A" in module B - a classical circular import, which comes with a lot of headaches and often results in overly convoluted code. All this could be simplified if it was possible to pass some module-level global variables to the about-to-be-imported module before its actual execution, but how the heck could that work with an import statement?

Later, I worked and experimented a lot in jupyter notebooks. Usually I would have some regular python module only with pure functions open in an editor, import that module in jupyter and have all the state there. Most of the time I'd spend in jupyter, but from time to time I'd edit the file and have to manually reload the module in jupyter - why couldn't it just automatically reload the module when my file is saved? Also, in the past reload() was a builtin function, now I have to import it extra from importlib, another hoop to jump through..

Then there were situations where I found a cool module on github, a single file in a package I wanted to import - but why is it necessary to download it manually or maybe pip install an outdated package? Shouldn't it be possible to import just *this exact file* and somehow make it secure enough that I don't have to worry about man-in-the-middle attacks or some github-hack?

On my journey I came across many blogposts, papers and presentation-notebooks with code that just 'import library' but mentions nowhere which actual version is being used, so  trying to copy and paste this code in order to reproduce the presented experiment or solution is doomed to failure.

I also remember how I had some code in a jupyter notebook that did 'import opencv' but I had not noted which actual version I had initially used. When I tried to run this notebook after a year again, it failed in a subtle way: the call signature of an opencv-function had slightly changed. It took quite a while to track down what my code was expecting and when this change occured until I figured out how to fix this issue. This could've been avoided or at least made a lot simpler if my imports were somehow annotated and checked in a pythonic way. After I complained about this in IRC, nedbat suggested an alternative functional way for imports with assignments: `mod = import("name", version)` which I found very alien and cumbersome at first sight - after all, we have an import statement for imports, and *there should be one and preferrably only one way to do it* - right?

Well, those shortcomings of the import statement kept bugging me. And when I stumbled over the word 'use' as a name for whatever I was conceiving, I thought "what the heck, let's try it!". Now use() can cover all my original usecases and there's even more to come!

# Examples

 import use
 
 np = use("numpy", version="1.19.2")
 
 use("pprint").pprint(some_dict)  
 
 tools = use(use.Path("/media/sf_Dropbox/code/tools.py"), reloading=True)
 
 test = use("functions", initial_globals={"foo":34, "bar":49})
 
 utils = use(use.URL("https://raw.githubusercontent.com/PIA-Group/BioSPPy/7696d682dc3aafc898cd9161f946ea87db4fed7f/biosppy/utils.py"),
            hash_value="95f98f25ef8cfa0102642ea5babbe6dde3e3a19d411db9164af53a9b4cdcccd8")

Thanks to the *default* keyword argument, it is also easy to simplify the rather clumsy optional import usecase like

```
try:
    import numpy as np
except ImportError:
    np = None
```
which would simply become
`np = use("numpy", default=None)`
while it is possible of course to pass anything as default, for instance some fallback that should be used instead if the preferred module/package isn't available. Note that you can cascade different use() this way! For instance, you can try to use() a local module with reload in a certain version, but if that fails fall back to a specific, reliably working version pulled from the web but that might not be the newest and best optimized.

## Beware of Magic!
Inspired by the q package/module, use() also is a callable class that replaces the module on import, so that only 'import use' is needed to be able to call use() on things, which might confuse users and libraries that operate on modules themselves.

Probably the most magical thing about use() is how automatic reload is realized. Whenever you `use(something, reloading=True)`, you won't get your actual module but a stand-in replacement, a so-called SurrogateModule instead. The actual module is imported whenever the file changed and the implementation is transparently replaced in the background. This way, you can keep references to the things in your module without problems, but thanks to this indirection it is possible that when you try to access something in your module, the current implementation is dynamically evaluated per call. Using an async loop, the file you specified as module is opened and hashed. Only if the hash has changed, it is actually attempted to execute and import the code. This means that if you properly imported a module at first but then edited and left a SyntaxError in, it will report this error when it tries to import the file again, but it won't replace the previous implementation until you edited the file and it could import it without error.
