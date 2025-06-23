import tempfile
from pathlib import Path

import requests


def test_read_wheel_metadata(reuse):
    content = requests.get(
        "https://files.pythonhosted.org/packages/45/80/cdf0df938fe63457f636d859499f4aab3d0411a90fd9472ad720a0b7eab6/justuse-0.5.0.tar.gz"
    ).content
    file = Path(tempfile.mkstemp(".tar.gz", "justuse-0.5.0")[1])
    file.write_bytes(content)
    whl_path = file
    if whl_path.exists():
        assert whl_path.exists()
        assert whl_path.is_file()
        meta = reuse.pimp.archive_meta(whl_path)
        assert meta
        assert meta["name"] == "justuse"
        assert meta["import_relpath"].endswith("use/__init__.py")
