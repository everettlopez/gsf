# src/clock.py
"""Simulation clock. The whole project reads 'now' from here, not datetime.now().

A run can last several workdays (main.py, GA_DAYS). The date moves on one day
per workday, so memories keep their real order and recency; the day's name
comes from WEEKDAYS, starting on the Thursday the personas' "yesterday"
(Wednesday) leads into, and skipping weekends.
"""
from datetime import datetime, timedelta

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
FIRST_DAY = "Thursday"

_now = datetime.combine(datetime.today().date(), datetime.strptime("7:00 AM", "%I:%M %p").time())
_day = 1                    # which workday of the run this is: 1, 2, ...


def set_time(time_str: str):
    """Set the sim time from a string like '7:05 am' (the date stays the same)."""
    global _now
    t = datetime.strptime(time_str.upper(), "%I:%M %p").time()
    _now = datetime.combine(_now.date(), t)


def now() -> datetime:
    return _now


def day() -> int:
    """The run's workday: 1 on the first day."""
    return _day


def day_name(number=None) -> str:
    """'Thursday' on day 1, 'Friday' on day 2, then 'Monday' (weekends off)."""
    start = WEEKDAYS.index(FIRST_DAY)
    return WEEKDAYS[(start + (number or _day) - 1) % len(WEEKDAYS)]


def next_day(time_str="7:00 am"):
    """The next workday starts: a day later, at time_str."""
    global _now, _day
    _day += 1
    _now = datetime.combine(_now.date() + timedelta(days=1),
                            datetime.strptime(time_str.upper(), "%I:%M %p").time())


def reset(time_str="7:00 am"):
    """Day 1 again, at time_str (a run's start; tests)."""
    global _day
    _day = 1
    set_time(time_str)

STEP_MINUTES = 5            # simulated minutes per step


def step_time(current_time, minutes=STEP_MINUTES):
    """'7:00 am' -> '7:05 am'. No zero-pad, matching the plan's format."""
    new_dt = datetime.strptime(current_time.upper(), "%I:%M %p") + timedelta(minutes=minutes)
    return new_dt.strftime("%I:%M %p").lstrip("0").lower()
