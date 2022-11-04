from datetime import datetime, timedelta


def datetime_now():
    current_datetime = datetime.now()
    return current_datetime, current_datetime.strftime("%d/%m/%Y %H:%M:%S")


def seconds_from_now(timestamp: datetime, seconds: int):
    return timestamp + timedelta(seconds=seconds) < datetime.now()
