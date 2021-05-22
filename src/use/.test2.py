from pathlib import Path

import use

mea = use(
    Path("F:\Dropbox (Privat)\mcs\Code\Arrhythmia_detection\hamamea\mea.py"),
    reloading=2,
)

mea.test()
