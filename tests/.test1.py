from threading import Thread
from types import ModuleType

class SurrogateModule(ModuleType):
    def __init__(self, spec, initial_globals):
        self.__implementation = ModuleType("")
        self.__stopped = False
        self.__thread = Thread(target=self.__reload)

        def __reload():
            last_filehash = None
            while not self._stopped:
                with open(spec.origin, "rb") as file:
                    current_filehash = hashfileobject(file)
                    if current_filehash != last_filehash:
                        try:
                            file.seek(0)
                            mod = build_mod(spec.name, file.read(), initial_globals)
                            self.__implementation = mod
                        except Exception as e:
                            print(e, traceback.format_exc())
                    last_filehash = current_filehash

    def __del__(self):
        self.__stopped = True

    def __getattribute__(self, name):
        if name in ("_SurrogateModule__reloading",
                     "_SurrogateModule__implementation"):
            return object.__getattribute__(self, name)
        else:
            return getattr(self.__implementation, name)
    
    def __setattr__(self, name, value):
        if name in ("_SurrogateModule__reloading",
                     "_SurrogateModule__implementation"):
            object.__setattr__(self, name, value)
        else:
            setattr(self.__implementation, name, value)