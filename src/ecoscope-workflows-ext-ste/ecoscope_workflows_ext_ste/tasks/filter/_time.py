import pandas as pd
from typing import Literal
from wt_registry import register
from pydantic import BaseModel, ConfigDict
from pandas.tseries.offsets import DateOffset
from dateutil.relativedelta import relativedelta
from ecoscope.platform.tasks.filter._filter import TimeRange

_SECONDS_PER_UNIT = {
    "seconds": 1,
    "minutes": 60,
    "hours": 3600,
    "days": 86400,
    "weeks": 604800,
}


class DateOffsetSchema(BaseModel):
    model_config = ConfigDict(title="DateOffset")
    n: int = 1
    years: int = 0
    months: int = 0
    weeks: int = 0
    days: int = 0
    hours: int = 0
    minutes: int = 0
    seconds: int = 0

    def to_offset(self) -> DateOffset:
        kwds = {k: v for k, v in self.model_dump().items() if k != "n" and v}
        return DateOffset(n=self.n, **kwds)

    @classmethod
    def from_offset(cls, off: DateOffset) -> "DateOffsetSchema":
        return cls(n=off.n, **off.kwds)


def _offset_description(n, years, months, weeks, days, hours, minutes, seconds) -> str:
    parts = [
        (years, "year"),
        (months, "month"),
        (weeks, "week"),
        (days, "day"),
        (hours, "hour"),
        (minutes, "minute"),
        (seconds, "second"),
    ]
    desc = ", ".join(f"{v} {u}{'s' if v != 1 else ''}" for v, u in parts if v)
    if n != 1 and desc:
        desc = f"{n}x ({desc})"
    return desc or "zero offset"


@register()
def create_date_offset(
    n: int = 1,
    years: int = 0,
    months: int = 0,
    weeks: int = 0,
    days: int = 0,
    hours: int = 0,
    minutes: int = 0,
    seconds: int = 0,
) -> DateOffsetSchema:
    desc = _offset_description(n, years, months, weeks, days, hours, minutes, seconds)
    print(f"[create_date_offset] Building offset: {desc}")
    result = DateOffsetSchema(
        n=n,
        years=years,
        months=months,
        weeks=weeks,
        days=days,
        hours=hours,
        minutes=minutes,
        seconds=seconds,
    )
    print(f"[create_date_offset] Offset ready: {desc}")
    return result


@register()
def shift_period(
    timerange: TimeRange,
    offset: DateOffsetSchema,
) -> TimeRange:
    desc = _offset_description(
        offset.n, offset.years, offset.months, offset.weeks, offset.days, offset.hours, offset.minutes, offset.seconds
    )
    print(f"[shift_period] Shifting {timerange.since} → {timerange.until} back by {desc}")
    off = offset.to_offset()
    since = (pd.Timestamp(timerange.since) - off).to_pydatetime()
    until = (pd.Timestamp(timerange.until) - off).to_pydatetime()
    result = TimeRange(
        since=since,
        until=until,
        timezone=timerange.timezone,
        time_format=timerange.time_format,
    )
    print(f"[shift_period] Shifted period: {result.since} → {result.until}")
    return result


@register()
def previous_period(
    timerange: TimeRange,
    offset: DateOffsetSchema,
) -> TimeRange:
    desc = _offset_description(
        offset.n, offset.years, offset.months, offset.weeks, offset.days, offset.hours, offset.minutes, offset.seconds
    )
    print(f"[previous_period] Computing period prior to {timerange.since}, stepping back {desc}")
    since = (pd.Timestamp(timerange.since) - offset.to_offset()).to_pydatetime()
    until = timerange.since
    result = TimeRange(
        since=since,
        until=until,
        timezone=timerange.timezone,
        time_format=timerange.time_format,
    )
    print(f"[previous_period] Previous period: {result.since} → {result.until}")
    return result


@register()
def get_duration(
    time_range: TimeRange,
    time_unit: Literal["seconds", "minutes", "hours", "days", "weeks", "months", "years"] = "months",
) -> float:
    since, until = time_range.since, time_range.until
    print(f"[get_duration] Measuring {since} to {until} in {time_unit}")
    # Fixed-length units
    if time_unit in _SECONDS_PER_UNIT:
        result = (until - since).total_seconds() / _SECONDS_PER_UNIT[time_unit]
        print(f"[get_duration] Duration: {result:.4f} {time_unit}")
        return result

    # Calendar units: whole units exact; partial unit measured against its real length.
    if time_unit in ("months", "years"):
        step = 12 if time_unit == "years" else 1
        rd = relativedelta(until, since)
        whole = rd.years * 12 + rd.months
        anchor = since + relativedelta(months=whole)
        next_anchor = since + relativedelta(months=whole + step)
        unit_len = (next_anchor - anchor).total_seconds()
        fraction = (until - anchor).total_seconds() / unit_len if unit_len else 0.0
        result = whole / step + fraction
        print(f"[get_duration] Duration: {result:.4f} {time_unit}")
        return result

    raise ValueError(f"`get_duration`: invalid time_unit {time_unit!r}")
