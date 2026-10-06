import datetime as dt
import holidays
import pandas as pd


def add_daytype(
    date: dt.datetime,
    holiday_list: tuple|None = None
) -> str:
    """
    Determines the type of day (e.g., Holiday, Saturday, Weekday) based on the provided date.

    Args:
        `date` (`datetime`): The date for which the day type is to be determined.
        `holiday_list` (`tuple`, optional): A tuple of strings representing holiday dates in the format `YYYY-MM-DD`.
            Defaults to a predefined list of holidays for the year 2024.

    Returns:
        `str`: The type of day: `'Holiday'`, `'Saturday'`, or `'Weekday'`.
    """
    if check_holiday(date):
        return 'Holiday'
    elif date.weekday() == 5:
        return 'Saturday'
    elif date.weekday() < 5:
        return 'Weekday'
    else:
        return 'Holiday'


def check_holiday(
    date: dt.datetime
) -> bool:
    it_holidays = holidays.country_holidays('IT', subdiv='BO', years=date.year)
    return date in it_holidays.keys()


def add_daytype_to_series(
    dates: pd.Series
) -> pd.Series:
    years = dates.dt.year.to_list()
    it_holidays = holidays.country_holidays('IT', subdiv='BO', years=range(min(years), max(years)+1))
    holiday_datetimes = pd.to_datetime(list(it_holidays))

    weekday_num = dates.dt.weekday

    is_holiday = dates.dt.normalize().isin(holiday_datetimes)
    is_saturday = weekday_num == 5
    is_weekday = weekday_num < 5

    day_type = pd.Series(index=dates.index, dtype="object")
    day_type[is_weekday & ~is_holiday] = "Weekday"
    day_type[is_saturday & ~is_holiday] = "Saturday"
    day_type[is_holiday | (weekday_num == 6)] = "Holiday"

    return day_type