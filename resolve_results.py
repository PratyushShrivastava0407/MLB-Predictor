#!/usr/bin/env python
"""
Fills in actual outcomes for logged picks (results/predictions_log.jsonl) whose
games have gone Final. Run this periodically (once a day is plenty) to keep the
log up to date.

For each unresolved pick, records:
  - actual_first_pa_pitches / actual_first_pa_over: the batter's FIRST plate
    appearance that game against the logged pitcher specifically.
  - actual_all_pa_pitches / actual_all_pa_over_count / actual_all_pa_total:
    every PA that batter had against that specific pitcher (a pitcher pulled
    mid-game means fewer PA than the batter's full game).
  - actual_slot_pitches / actual_slot_over: the outcome of the SPECIFIC PA
    that matches the pick's own pa_slot -- e.g. a pick logged with pa_slot='2'
    is checked against the batter's actual 2nd PA of the game (counting all
    his PAs that game, any pitcher, same definition as
    mlbtool.metrics.pa_num_this_game), not just his first meeting with that
    pitcher. None if the batter never reached that slot (e.g. pulled from the
    game, or the matching pitcher was already out by then).

All three are recorded because we don't know which framing a given
sportsbook's market actually resolves on (see the WSH@NYM conversation --
first-PA and all-PA results can diverge a lot), and actual_slot_* is what
actually validates whether a given --pa-slot pick was accurate. Batches by
game_pk so a game with multiple logged picks only costs one API call.

Usage: python resolve_results.py [--summary]
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import requests

from mlbtool import results_log

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

BASE_1_1 = "https://statsapi.mlb.com/api/v1.1"


def game_is_final(game_pk: int) -> bool:
    r = requests.get(f"{BASE_1_1}/game/{game_pk}/feed/live", timeout=20)
    r.raise_for_status()
    data = r.json()
    return data["gameData"]["status"]["abstractGameState"] == "Final", data


def _legacy_slot(entry: dict) -> str:
    """Migrate pre-pa_slot log entries (which only had a binary pa_mode field)."""
    if entry.get("pa_slot"):
        return entry["pa_slot"]
    mode = entry.get("pa_mode")
    return "1" if mode == "first" else "all"


def resolve_game(entries_for_game: list[dict], live_data: dict) -> None:
    plays = live_data.get("liveData", {}).get("plays", {}).get("allPlays", [])

    # Running PA count per batter across the whole game (any pitcher) -- same
    # definition as pa_num_this_game in mlbtool.metrics, computed independently
    # here since we're working from the live feed, not a Statcast pull.
    batter_pa_count = defaultdict(int)
    play_pa_num = []
    for p in plays:
        bname = p["matchup"]["batter"]["fullName"]
        batter_pa_count[bname] += 1
        play_pa_num.append(batter_pa_count[bname])

    for e in entries_for_game:
        match_idxs = [
            i for i, p in enumerate(plays)
            if p["matchup"]["batter"]["fullName"] == e["batter_name"]
            and p["matchup"]["pitcher"]["fullName"] == e["opposing_pitcher"]
        ]
        if not match_idxs:
            continue  # batter never faced this pitcher (pulled early, scratched, etc.)

        pitch_counts = [len(plays[i].get("playEvents", [])) for i in match_idxs]
        e["actual_first_pa_pitches"] = pitch_counts[0]
        e["actual_first_pa_over"] = pitch_counts[0] >= 4
        e["actual_all_pa_pitches"] = pitch_counts
        e["actual_all_pa_over_count"] = sum(1 for n in pitch_counts if n >= 4)
        e["actual_all_pa_total"] = len(pitch_counts)

        slot = _legacy_slot(e)
        e["pa_slot"] = slot  # backfill so older entries carry the field going forward
        if slot == "all":
            e["actual_slot_pitches"] = None
            e["actual_slot_over"] = None
        else:
            want = 4 if slot == "4+" else int(slot)
            hit_idx = next(
                (i for i in match_idxs if (play_pa_num[i] >= 4 if slot == "4+" else play_pa_num[i] == want)),
                None,
            )
            if hit_idx is not None:
                n = len(plays[hit_idx].get("playEvents", []))
                e["actual_slot_pitches"] = n
                e["actual_slot_over"] = n >= 4
            else:
                # Batter never reached this PA slot against this pitcher that game
                # (e.g. the pitcher was pulled before the batter's 2nd/3rd/4th look).
                e["actual_slot_pitches"] = None
                e["actual_slot_over"] = None

        e["resolved"] = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true", help="Print hit-rate summary after resolving")
    args = ap.parse_args()

    entries = results_log.load_all()
    if not entries:
        print("No logged picks yet -- run predict_game.py first.")
        return

    pending = [e for e in entries if not e["resolved"]]
    by_game = defaultdict(list)
    for e in pending:
        by_game[e["game_pk"]].append(e)

    print(f"{len(pending)} pending pick(s) across {len(by_game)} game(s).")
    resolved_count = 0
    for game_pk, game_entries in by_game.items():
        try:
            is_final, live_data = game_is_final(game_pk)
        except Exception as ex:
            print(f"  gamePk {game_pk}: fetch failed ({ex}), skipping")
            continue
        if not is_final:
            print(f"  gamePk {game_pk}: not Final yet, skipping")
            continue
        resolve_game(game_entries, live_data)
        newly = sum(1 for e in game_entries if e["resolved"])
        resolved_count += newly
        print(f"  gamePk {game_pk}: resolved {newly}/{len(game_entries)} pick(s)")

    results_log.save_all(entries)
    print(f"\nResolved {resolved_count} new pick(s). Log saved to {results_log.LOG_PATH}")

    if args.summary:
        print_summary(entries)


def aligned_hit(e: dict):
    """
    Judge a pick against the outcome its own pa_slot actually targeted:
      - pa_slot='all' -> the blended all-PA-vs-that-pitcher rate (0.0-1.0).
      - pa_slot in {'1','2','3','4+'} -> 1.0/0.0 from actual_slot_over, the
        specific PA that matches that slot.
    Returns None if that slot never happened for this matchup that game (e.g.
    the pitcher was pulled before the batter's 2nd/3rd look) -- these are
    excluded from hit-rate math rather than counted as a loss, since a real
    book would likely grade a bet on a PA that never occurred as no-action.
    """
    slot = e.get("pa_slot", "all")
    if slot == "all":
        total = e["actual_all_pa_total"] or 0
        return (e["actual_all_pa_over_count"] or 0) / total if total else None
    if e.get("actual_slot_over") is None:
        return None
    return 1.0 if e["actual_slot_over"] else 0.0


def print_summary(entries: list[dict]) -> None:
    resolved = [e for e in entries if e["resolved"] and e["actual_first_pa_pitches"] is not None]
    if not resolved:
        print("\nNo resolved picks with outcomes yet.")
        return

    print("\n" + "=" * 70)
    print(" SUMMARY -- resolved picks")
    print("=" * 70)

    def bucket_stats(rows, label_fn):
        buckets = defaultdict(lambda: {
            "n": 0, "n_no_data": 0, "aligned_n": 0, "aligned_hits": 0.0,
            "first_pa_over": 0, "all_pa_over": 0, "all_pa_total": 0,
        })
        for e in rows:
            b = buckets[label_fn(e)]
            b["n"] += 1
            b["first_pa_over"] += 1 if e["actual_first_pa_over"] else 0
            b["all_pa_over"] += e["actual_all_pa_over_count"] or 0
            b["all_pa_total"] += e["actual_all_pa_total"] or 0
            h = aligned_hit(e)
            if h is None:
                b["n_no_data"] += 1
            else:
                b["aligned_n"] += 1
                b["aligned_hits"] += h
        return buckets

    def fmt_bucket(label, b):
        fp_rate = b["first_pa_over"] / b["n"] * 100 if b["n"] else 0
        ap_rate = b["all_pa_over"] / b["all_pa_total"] * 100 if b["all_pa_total"] else 0
        aligned_rate = b["aligned_hits"] / b["aligned_n"] * 100 if b["aligned_n"] else None
        aligned_str = f"{aligned_rate:.1f}%" if aligned_rate is not None else "n/a"
        no_data_str = f", {b['n_no_data']} no-data" if b["n_no_data"] else ""
        print(f"  {label:<8} n={b['n']:<4} aligned hit rate: {aligned_str}{no_data_str}   "
              f"(first-PA vs pitcher: {b['first_pa_over']}/{b['n']} [{fp_rate:.1f}%]   "
              f"all-PA vs pitcher: {b['all_pa_over']}/{b['all_pa_total']} [{ap_rate:.1f}%])")

    print("\nBy confidence tier:")
    for conf, b in sorted(bucket_stats(resolved, lambda e: e["confidence"]).items()):
        fmt_bucket(conf, b)

    print("\nBy PA slot:")
    for slot, b in sorted(bucket_stats(resolved, lambda e: e.get("pa_slot", "all")).items()):
        fmt_bucket(slot, b)

    overall = bucket_stats(resolved, lambda e: "all")["all"]
    avg_pred = sum(e["predicted_probability"] for e in resolved) / len(resolved)
    aligned_rate = overall["aligned_hits"] / overall["aligned_n"] * 100 if overall["aligned_n"] else 0
    print(f"\nOverall: n={len(resolved)}   avg predicted P(4+)={avg_pred*100:.1f}%   "
          f"aligned hit rate={aligned_rate:.1f}% (n={overall['aligned_n']}, "
          f"{overall['n_no_data']} excluded as no-data) "
          f"-- each pick judged against the outcome its own pa_slot targeted")


if __name__ == "__main__":
    main()
