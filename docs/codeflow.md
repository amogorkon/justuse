This is an overflow of how justuse works. This should give you an idea where to look for things. Don't rely on this document.

## Initialization

```mermaid
graph LR;
    A(__init__.py)-->B(main.py);
    B(main.py)-->C(instance of class Use replaces module use in __init__.py);
```
## Modes of operation

This is defined on the class Use via singledispatch on the type of argument
```mermaid
graph LR;
    A(use) --> B[\Path - module is a local file\]
    A(use) --> C[\URL - module is an online resource\]
    A(use) --> D[\string - module is part of a package, which may or may not be installed\]
```

## use package
While using local or online resources as modules is very straight forward, using modules as part of packages is not.
To use a module from a package with auto-installation, you need to think of the name to be used as two-parted. The first part is how you would pip-install it, the second is how you would import the module. You can specify the name in three ways:

    # package name that you would pip install = "py.foo"
    # module name that you would import = "foo.bar"
    
    use("py.foo/foo.bar")
    use(("py.foo", "foo.bar"))
    use(package_name="py.foo", module_name="foo.bar")

However you call it, it all gets normalized into `name` (the single string representation), `package_name` for installation and `module_name` for import.

```mermaid
graph TD
    A(main.Use._use_package) --> B[\normalization of all call-data into a dictionary of keyword args\]
    B --> C["buffet table" of functions, for dispatch on whichever combination of conditions]
    C --> D[\each function gets called with the same kwargs, picking the kwargs it needs, ignoring the rest\]
```

### auto-installation
Inline-installation of packages is one of the most interesting and complex features of justuse.

With version and hash properly defined and auto-installation requested, the flow of action is as following:
```mermaid
graph TD
    A(main.Use._use_package) --> B[\pimp._auto_install\]
    B --> C{found in registry?}
    C -- yes --> D{zip?}
    D -- yes --> E[\try to import it directly via zipimport\]
    D -- no --> F[\try to install it using pip\]
    C -- no --> G[\download the artifact\]
    G --> D
    F --> H[import it via importlib]
    H --> I
    E --> I[return mod]
```
