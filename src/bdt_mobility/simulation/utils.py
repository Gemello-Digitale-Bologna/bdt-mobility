import pandas as pd

def to_time(seconds):
    return pd.Timestamp('00:00:00').to_pydatetime() + pd.Timedelta(seconds=seconds)

def to_number(time):
    return (time - pd.Timestamp('00:00:00')).total_seconds()

