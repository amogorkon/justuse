from enum import Enum

class UserMessage(Enum):
    def not_reloadable(self, name):
        return f"Beware {name} also contains non-function objects, it may not be safe to reload!"