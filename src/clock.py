# src/clock.py
"""Simulation clock. The whole project reads 'now' from here, not datetime.now()."""
from datetime import datetime

_now = datetime.combine(datetime.today().date(), datetime.strptime("7:00 AM", "%I:%M %p").time())


def set_time(time_str: str):
    """Set the sim time from a string like '7:05 am' (the date stays the same)."""
    global _now
    t = datetime.strptime(time_str.upper(), "%I:%M %p").time()
    _now = datetime.combine(_now.date(), t)


def now() -> datetime:
    return _now