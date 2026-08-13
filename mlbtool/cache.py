"""
Lightweight on-disk cache so same-day reruns don't re-hit MLB Stats API / Baseball
Savant. Two tiers:

  - JSON cache (small lookups: game info, player ids/hand, lineups) with a TTL in
    minutes. Lineups get a short TTL because they can flip from "projected" to
    "confirmed" as game time approaches; static lookups (team ids, player ids) get
    a long TTL since they never change intraday.
  - Parquet cache (statcast pitch logs) keyed by role+player_id, valid for the
    calendar day it was pulled on. These are the expensive calls (multi-second HTTP
    pulls per player), so we never re-pull twice in the same day regardless of how
    many times the script is invoked.

Nothing here talks to the network -- callers pass in a "fetch" function that runs
only on a cache miss.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

CACHE_ROOT = Path(__file__).resolve().parent.parent / "cache"
JSON_DIR = CACHE_ROOT / "statsapi"
PARQUET_DIR = CACHE_ROOT / "statcast"

JSON_DIR.mkdir(parents=True, exist_ok=True)
PARQUET_DIR.mkdir(parents=True, exist_ok=True)


def _today_str() -> str:
    return time.strftime("%Y-%m-%d")


def json_cache(key: str, ttl_minutes: float, fetch: Callable[[], dict]) -> dict:
    """Return cached JSON for `key` if younger than ttl_minutes, else call fetch()."""
    path = JSON_DIR / f"{key}.json"
    if path.exists():
        age_minutes = (time.time() - path.stat().st_mtime) / 60.0
        if age_minutes <= ttl_minutes:
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass  # fall through and refetch on a corrupt cache file
    data = fetch()
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return data


def statcast_cache(role: str, player_id: int, fetch: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    """
    Return cached statcast pitch log for (role, player_id) if it was pulled today,
    else call fetch() and persist. `role` is 'pitcher' or 'batter'.
    """
    base = PARQUET_DIR / f"{role}_{player_id}"
    data_path = base.with_suffix(".parquet")
    meta_path = base.with_suffix(".meta.json")

    if data_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("pulled_on") == _today_str():
                return pd.read_parquet(data_path)
        except (json.JSONDecodeError, OSError):
            pass

    df = fetch()
    try:
        df.to_parquet(data_path, index=False)
        meta_path.write_text(
            json.dumps({"pulled_on": _today_str(), "rows": len(df)}), encoding="utf-8"
        )
    except Exception:
        pass  # cache write failures shouldn't break a run
    return df


def cache_age_note(role: str, player_id: int) -> Optional[str]:
    """Human-readable note on when this player's statcast cache was last refreshed."""
    meta_path = PARQUET_DIR / f"{role}_{player_id}.meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return meta.get("pulled_on")
    except (json.JSONDecodeError, OSError):
        return None
