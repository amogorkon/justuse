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