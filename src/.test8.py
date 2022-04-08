import inspect
from typing import Any, get_args, get_origin
from numbers import Number, Complex, Real, Rational, Integral
from collections.abc import Sequence
from itertools import zip_longest
from warnings import warn
import typing
import collections


def test1(a, b, *, y:float, x:str):
    return x

def test2(a, b, *, x: str, y:float):
    return x

def test3(x: float):
    return x

def test4(x: int):
    return x

def test5(x: Sequence[int]):
    return x

def test6(x: list[int]):
    return x

def test7(x: typing.Sequence):
    return x

def test8(x: Sequence):
    return x

numerics = [bool, Integral, Rational, Real, Complex]

class NotComparableWarning(UserWarning):
    pass

def is_compatible(pre, post):
    sig = inspect.signature(pre)
    pre_sig = []
    # first separate args and kwargs so we can sort kwargs alphabetically
    args = []
    kwargs = []
    
    for k, v in sig.parameters.items():
        if v.kind is v.VAR_KEYWORD or v.POSITIONAL_OR_KEYWORD:
            kwargs.append((k,v))
        else:
            args.append((k,v))
    
    for k, v in args:
        v = v.annotation
        pre_sig.append(v if v is not inspect._empty else Any)
    
    for k, v in sorted(kwargs):
        v = v.annotation
        pre_sig.append(v if v is not inspect._empty else Any)
        
    v = sig.return_annotation
    pre_sig.append(v if v is not inspect._empty else Any)
    
    sig = inspect.signature(post)
    post_sig = []
    
    args = []
    kwargs = []
    
    for k, v in sig.parameters.items():
        if v.kind is v.VAR_KEYWORD or v.POSITIONAL_OR_KEYWORD:
            kwargs.append((k,v))
        else:
            args.append((k,v))
    
    for k, v in args:
        v = v.annotation
        post_sig.append(v if v is not inspect._empty else Any)
    
    for k, v in sorted(kwargs):
        v = v.annotation
        post_sig.append(v if v is not inspect._empty else Any)
    
    v = sig.return_annotation
    post_sig.append(v if v is not inspect._empty else Any)
        
    def check(x, y):
        print(f"checking {x} {type(x)} => {y} {type(y)}")
        # narrowing {(Any => Any) | (Any => something)} is OK
        if x is Any:
            return True
        else:
            # broadening (something => Any) is NOK
            if y is Any:
                return False
            # now to the more specific cases (something => something else)
            else:
                # let's first check if we're dealing with numbers
                try:
                    if issubclass(x, Number) and issubclass(y, Number):
                        # since the generic implementations aren't actual subclasses of each other
                        # we have to map to the numeric tower classes
                        for X in numerics:
                            if issubclass(x, X):
                                break
                        x = X
                        for Y in numerics:
                            if issubclass(y, Y):
                                break
                        y = Y
                except TypeError:  # issubclass is allergic to container classes (types.GenericAlias)
                    pass
                
                # the other important type hierarchy are containers
                # let's check if the user is using typing classes and educate them to use collections.abc instead
                if x.__module__ == "typing":
                    x = getattr(collections.abc, str(x).split(".")[-1])
                    warn(NotComparableWarning(f"{x} is of a type from the typing module, which can't be compared. Please use a type from collections.abc instead.")) 
                if y.__module__ == "typing":
                    y = getattr(collections.abc, str(y).split(".")[-1])
                    warn(NotComparableWarning(f"{y} is of a type from the typing module, which can't be compared. Please use a type from collections.abc instead.")) 
                
                X = get_origin(x)
                Y = get_origin(y)
                if X is not None and Y is not None:
                    # (Sequence => list) is OK, (list => Sequence) is NOK
                    if issubclass(Y, X):
                        # (list[int] => Sequence[int]) -> (int => int)
                        return all(check(x_, y_) for x_, y_ in zip_longest(get_args(x), get_args(y), fillvalue=Any))
                    else:
                        return False
                    
                try:
                    # (x => y) where y is not a subclass of x is NOK
                    return issubclass(y, x)
                except TypeError:  # (int => list[int]) or (list[int] => int) NOK
                    return False

    return all(check(x, y) for x, y in zip_longest(pre_sig, post_sig, fillvalue=Any))
        
#print(is_compatible(test1, test2))
#print(is_compatible(test5, test6))
print(is_compatible(test1, test2))
