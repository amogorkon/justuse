from pathlib import Path
from hashlib import sha256

p = Path(r"C:\Users\micro\.justuse-python\web-modules\20230325.368646_flux.pyc")

algo = sha256
content = p.read_bytes()
print(algo(content).hexdigest())