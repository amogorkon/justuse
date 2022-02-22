from logging import DEBUG, basicConfig, getLogger

from pathlib import Path
home = Path.home() / ".justuse-python"
from datetime import datetime
def fraction_of_day(now: datetime = None) -> float:
    if now is None:
        now = datetime.utcnow()
    return round(
        (
            now.hour / 24
            + now.minute / (24 * 60)
            + now.second / (24 * 60 * 60)
            + now.microsecond / (24 * 60 * 60 * 1000 * 1000)
        )
        * 1000,
        6,
    )

basicConfig(
    filename=home / "logs" / "usage.log",
    filemode="a",
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt=f"%Y%m%d {fraction_of_day()}",
    # datefmt="%Y-%m-%d %H:%M:%S",
    level=DEBUG,
)


log = getLogger("adsf")

log.critical("foobar!")