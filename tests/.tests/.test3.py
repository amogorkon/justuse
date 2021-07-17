import datetime
# calculate seconds since beginning of the year
def seconds_since_beginning_of_year(dt: datetime) -> int:
        return (dt - datetime.datetime(dt.year, 1, 1)).total_seconds()

print(seconds_since_beginning_of_year(datetime.datetime.now()))