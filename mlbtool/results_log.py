"""
Persistent log of every pick the model has recommended, so hit rates can be
checked against reality later instead of relying on one-off spot checks in
conversation. Two halves:

  - log_picks(): called automatically at the end of every predict_game.py run,
    appends one JSON line per recommended pick (unresolved).
  - resolve_pending(): run separately (resolve_results.py) once games are
    Final -- pulls each game's actual play-by-play, finds the logged batter's
    PA(s) against the logged pitcher, and fills in what actually happened.

Kept as append-only JSONL, not sqlite -- there's no query pattern here that
needs an index, and a flat file is trivial to inspect/diff/hand-edit.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
LOG_PATH = RESULTS_DIR / "predictions_log.jsonl"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def log_picks(game_pk: int, game_date: str, team1: str, team2: str, results: list) -> int:
    """results: list of (MatchupResult, bullets) tuples already filtered to the
    ones actually recommended. Returns how many were logged."""
    if not results:
        return 0
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    n = 0
    with LOG_PATH.open("a", encoding="utf-8") as f:
        for r, _bullets in results:
            entry = {
                "logged_at": now,
                "game_date": game_date,
                "game_pk": game_pk,
                "team1": team1,
                "team2": team2,
                "batter_name": r.batter_name,
                "batter_id": r.batter_id,
                "batter_team": r.team_abbr,
                "opposing_pitcher": r.pitcher_name,
                "batting_order": r.order,
                "predicted_probability": round(r.probability, 4),
                "edge": round(r.edge, 4),
                "confidence": r.confidence,
                "resolved": False,
                "actual_first_pa_pitches": None,
                "actual_first_pa_over": None,
                "actual_all_pa_pitches": None,
                "actual_all_pa_over_count": None,
                "actual_all_pa_total": None,
            }
            f.write(json.dumps(entry) + "\n")
            n += 1
    return n


def load_all() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    entries = []
    with LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def save_all(entries: list[dict]) -> None:
    with LOG_PATH.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def pending_game_pks(entries: Optional[list[dict]] = None) -> set:
    entries = entries if entries is not None else load_all()
    return {e["game_pk"] for e in entries if not e["resolved"]}
