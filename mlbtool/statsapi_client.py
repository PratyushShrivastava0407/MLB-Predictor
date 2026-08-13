"""
MLB Stats API client: team ids, today's game lookup, starting lineups (confirmed
when posted, otherwise a "most recent lineup used" fallback), and player id /
handedness lookups.

No API key needed -- statsapi.mlb.com is the public-ish endpoint the mlb.com
gameday pages themselves call.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional

import requests

from . import cache

BASE = "https://statsapi.mlb.com/api/v1"
BASE_1_1 = "https://statsapi.mlb.com/api/v1.1"
TIMEOUT = 20


@dataclass
class Player:
    id: int
    name: str
    bats: Optional[str] = None  # 'L' / 'R' / 'S'
    throws: Optional[str] = None


@dataclass
class LineupSlot:
    order: int  # 1-9
    player: Player


@dataclass
class GameInfo:
    game_pk: int
    game_date_utc: str
    home_abbr: str
    away_abbr: str
    home_team_id: int
    away_team_id: int


def _get(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def team_abbr_to_id() -> dict:
    def fetch():
        data = _get(f"{BASE}/teams", {"sportId": 1})
        return {t["abbreviation"]: t["id"] for t in data["teams"]}

    return cache.json_cache("team_abbr_to_id", ttl_minutes=60 * 24 * 7, fetch=fetch)


def find_game(date: str, team1_abbr: str, team2_abbr: str) -> GameInfo:
    """Find the game on `date` between the two teams (order-independent)."""
    key = f"schedule_{date}"

    def fetch():
        data = _get(
            f"{BASE}/schedule",
            {"sportId": 1, "date": date, "hydrate": "team"},
        )
        games = []
        for d in data.get("dates", []):
            games.extend(d.get("games", []))
        return {"games": games}

    data = cache.json_cache(key, ttl_minutes=30, fetch=fetch)
    wanted = {team1_abbr.upper(), team2_abbr.upper()}
    for g in data["games"]:
        away = g["teams"]["away"]["team"]["abbreviation"]
        home = g["teams"]["home"]["team"]["abbreviation"]
        if {away, home} == wanted:
            return GameInfo(
                game_pk=g["gamePk"],
                game_date_utc=g["gameDate"],
                home_abbr=home,
                away_abbr=away,
                home_team_id=g["teams"]["home"]["team"]["id"],
                away_team_id=g["teams"]["away"]["team"]["id"],
            )
    raise ValueError(
        f"No game found between {team1_abbr} and {team2_abbr} on {date}. "
        f"Games found that date: "
        + ", ".join(f"{g['teams']['away']['team']['abbreviation']}@{g['teams']['home']['team']['abbreviation']}" for g in data["games"])
    )


def lookup_player(name: str) -> Player:
    """Resolve a full name (e.g. 'Dylan Cease') to id + handedness via MLB search."""

    def fetch():
        data = _get(f"{BASE}/people/search", {"names": name})
        people = data.get("people", [])
        if not people:
            return {}
        # Prefer an exact case-insensitive full-name match; else first hit.
        exact = [p for p in people if p["fullName"].lower() == name.lower()]
        p = exact[0] if exact else people[0]
        return {
            "id": p["id"],
            "name": p["fullName"],
            "bats": p.get("batSide", {}).get("code"),
            "throws": p.get("pitchHand", {}).get("code"),
        }

    key = "player_" + "".join(c if c.isalnum() else "_" for c in name.lower())
    data = cache.json_cache(key, ttl_minutes=60 * 24 * 7, fetch=fetch)
    if not data:
        raise ValueError(f"Could not find MLB player named '{name}' via statsapi search.")
    return Player(id=data["id"], name=data["name"], bats=data.get("bats"), throws=data.get("throws"))


def people_batch(ids: list[int]) -> dict[int, Player]:
    """Batch-resolve handedness for a list of player ids (used for lineup players)."""
    ids = sorted(set(ids))
    if not ids:
        return {}
    key = "people_" + "_".join(str(i) for i in ids)

    def fetch():
        data = _get(f"{BASE}/people", {"personIds": ",".join(str(i) for i in ids)})
        out = {}
        for p in data.get("people", []):
            out[str(p["id"])] = {
                "id": p["id"],
                "name": p["fullName"],
                "bats": p.get("batSide", {}).get("code"),
                "throws": p.get("pitchHand", {}).get("code"),
            }
        return out

    data = cache.json_cache(key, ttl_minutes=60 * 24 * 7, fetch=fetch)
    return {
        int(pid): Player(id=v["id"], name=v["name"], bats=v.get("bats"), throws=v.get("throws"))
        for pid, v in data.items()
    }


def _extract_starters_from_boxscore(box_teams: dict, team_key: str) -> list[tuple[int, str, int]]:
    """Return [(battingOrder//100, name, id), ...] sorted, starters only (order % 100 == 0)."""
    players = box_teams[team_key]["players"]
    starters = []
    for _, p in players.items():
        bo = p.get("battingOrder")
        if bo is None:
            continue
        bo = int(bo)
        if bo % 100 == 0:
            starters.append((bo // 100, p["person"]["fullName"], p["person"]["id"]))
    starters.sort(key=lambda t: t[0])
    return starters


def get_confirmed_lineup(game_pk: int, side: str) -> Optional[list[LineupSlot]]:
    """
    side: 'home' or 'away'. Returns the confirmed starting lineup for this exact
    game if MLB has posted it (posted ~1-3 hours before first pitch), else None.
    """

    def fetch():
        data = _get(f"{BASE_1_1}/game/{game_pk}/feed/live", {})
        try:
            box = data["liveData"]["boxscore"]["teams"]
        except KeyError:
            return {"starters": []}
        starters = _extract_starters_from_boxscore(box, side)
        return {"starters": [{"order": o, "name": n, "id": i} for o, n, i in starters]}

    key = f"lineup_{game_pk}_{side}"
    data = cache.json_cache(key, ttl_minutes=15, fetch=fetch)
    starters = data.get("starters", [])
    if not starters:
        return None
    ids = [s["id"] for s in starters]
    people = people_batch(ids)
    return [
        LineupSlot(order=s["order"], player=people.get(s["id"], Player(id=s["id"], name=s["name"])))
        for s in starters
    ]


def get_fallback_lineup(team_id: int, before_date: str) -> tuple[list[LineupSlot], str]:
    """
    No confirmed lineup yet -> use the starting lineup from this team's most
    recent completed game before `before_date`. Returns (lineup, source_game_date).
    """
    end = dt.date.fromisoformat(before_date)
    start = end - dt.timedelta(days=12)

    def fetch():
        data = _get(
            f"{BASE}/schedule",
            {
                "sportId": 1,
                "teamId": team_id,
                "startDate": start.isoformat(),
                "endDate": (end - dt.timedelta(days=1)).isoformat(),
            },
        )
        games = []
        for d in data.get("dates", []):
            for g in d["games"]:
                if g["status"]["detailedState"] == "Final":
                    games.append({"gamePk": g["gamePk"], "date": d["date"]})
        return {"games": games}

    key = f"recent_games_{team_id}_{before_date}"
    data = cache.json_cache(key, ttl_minutes=60 * 6, fetch=fetch)
    games = data["games"]
    if not games:
        raise ValueError(f"No recent completed games found for team {team_id} to build a fallback lineup.")
    games.sort(key=lambda g: g["date"])
    last = games[-1]
    game_pk = last["gamePk"]

    def fetch_box():
        data = _get(f"{BASE_1_1}/game/{game_pk}/feed/live", {})
        box = data["liveData"]["boxscore"]["teams"]
        side = "home" if box["home"]["team"]["id"] == team_id else "away"
        starters = _extract_starters_from_boxscore(box, side)
        return {"starters": [{"order": o, "name": n, "id": i} for o, n, i in starters]}

    box_key = f"box_{game_pk}_{team_id}"
    box_data = cache.json_cache(box_key, ttl_minutes=60 * 24 * 30, fetch=fetch_box)
    starters = box_data["starters"]
    ids = [s["id"] for s in starters]
    people = people_batch(ids)
    lineup = [
        LineupSlot(order=s["order"], player=people.get(s["id"], Player(id=s["id"], name=s["name"])))
        for s in starters
    ]
    return lineup, last["date"]


def get_lineup(game_pk: int, team_id: int, side: str, date: str) -> tuple[list[LineupSlot], bool, Optional[str]]:
    """
    Returns (lineup, is_confirmed, fallback_source_date).
    Tries confirmed lineup first, falls back to most recent lineup used.
    """
    confirmed = get_confirmed_lineup(game_pk, side)
    if confirmed:
        return confirmed, True, None
    lineup, source_date = get_fallback_lineup(team_id, date)
    return lineup, False, source_date
