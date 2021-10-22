import tarfile
from pathlib import Path


class TarFunctions:
    def __init__(self, artifact_path):
        self.archive = tarfile.open(artifact_path)

    def get(self):
        return (
            self.archive,
            [m.name for m in self.archive.getmembers() if m.type == b"0"],
        )

    def read_entry(self, entry_name):
        m = self.archive.getmember(entry_name)
        with self.archive.extractfile(m) as f:
            bdata = f.read()
            text = bdata.decode("ISO-8859-1") if len(bdata) < 8192 else ""
            return (Path(entry_name).stem, text.splitlines())
