from pathlib import Path
import numpy as np

path = Path(r"F:\Dropbox (Privat)\mcs\Code\Arrhythmia_detection\annotator\results.npy")

results = np.load(path, allow_pickle=True).item()
print(results)