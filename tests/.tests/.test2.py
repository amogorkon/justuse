from enum import Enum

Modes = Enum("Modes", "a b c")

class Use:
    pass

for member in Modes:
    setattr(Use, member.name, member.value)
