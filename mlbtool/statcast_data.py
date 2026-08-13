"""
Pulls pitch-level Statcast history for a player via pybaseball, cached to disk
(see mlbtool.cache). We pull the current season plus the two prior seasons
rather than a full career -- for a 10+ year veteran, full-career Statcast data
is (a) slow to pull, (b) partly irrelevant (pitch mix, stuff, and approach
drift year to year), and (c) not what "career" should mean for a same-day bet.
We label this window "3yr" everywhere instead of "career" so the output isn't
overclaiming what it's based on.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pybaseball as pb

from . import cache

pb.cache.enable()  # pybaseball's own on-disk cache, belt-and-suspenders with ours

SEASONS_BACK = 2  # plus current season = 3 seasons total


def _window_start(today: dt.date) -> str:
    return f"{today.year - SEASONS_BACK}-01-01"


def pull_pitcher(player_id: int, as_of_date: str) -> pd.DataFrame:
    today = dt.date.fromisoformat(as_of_date)
    start = _window_start(today)

    def fetch():
        df = pb.statcast_pitcher(start, as_of_date, player_id)
        return df if df is not None else pd.DataFrame()

    return cache.statcast_cache("pitcher", player_id, fetch)


def pull_batter(player_id: int, as_of_date: str) -> pd.DataFrame:
    today = dt.date.fromisoformat(as_of_date)
    start = _window_start(today)

    def fetch():
        df = pb.statcast_batter(start, as_of_date, player_id)
        return df if df is not None else pd.DataFrame()

    return cache.statcast_cache("batter", player_id, fetch)
