from datetime import datetime


def test_fraction_of_day(reuse):
    assert reuse.fraction_of_day(datetime(2020, 1, 1)) == 0.0
    assert reuse.fraction_of_day(datetime(2020, 1, 1, 12)) == 500.0
    assert reuse.fraction_of_day(datetime(2020, 1, 1, 18)) == 750.0
    assert reuse.fraction_of_day(datetime(2020, 1, 1, 12, 30, 30)) == 521.180556
    assert reuse.fraction_of_day(datetime(2020, 1, 1, 12, 30, 30, 90000)) == 521.181597
