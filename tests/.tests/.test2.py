import typing

class list(list):
    @staticmethod
    def __class_getitem__(key):
         return typing.List[key]

        
print(hash(list))