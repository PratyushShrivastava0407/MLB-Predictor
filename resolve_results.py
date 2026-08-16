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

Both are recorded because we don't know which framing a given sportsbook's
market actually resolves on (see the WSH@NYM conversation -- first-PA and
all-PA results can diverge a lot). Batches by game_pk so a game with multiple
logged picks only costs one API call.

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


def resolve_game(entries_for_game: list[dict], live_data: dict) -> None:
    plays = live_data.get("liveData", {}).get("plays", {}).get("allPlays", [])
    for e in entries_for_game:
        matches = [
            p for p in plays
            if p["matchup"]["batter"]["fullName"] == e["batter_name"]
            and p["matchup"]["pitcher"]["fullName"] == e["opposing_pitcher"]
        ]
        if not matches:
            continue  # batter never faced this pitcher (pulled early, scratched, etc.)

        pitch_counts = [len(p.get("playEvents", [])) for p in matches]
        e["actual_first_pa_pitches"] = pitch_counts[0]
        e["actual_first_pa_over"] = pitch_counts[0] >= 4
        e["actual_all_pa_pitches"] = pitch_counts
        e["actual_all_pa_over_count"] = sum(1 for n in pitch_counts if n >= 4)
        e["actual_all_pa_total"] = len(pitch_counts)
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


def print_summary(entries: list[dict]) -> None:
    resolved = [e for e in entries if e["resolved"] and e["actual_first_pa_pitches"] is not None]
    if not resolved:
        print("\nNo resolved picks with outcomes yet.")
        return

    print("\n" + "=" * 70)
    print(" SUMMARY -- resolved picks")
    print("=" * 70)

    def aligned_hit(e: dict) -> float:
        """Judge each pick against the outcome its own pa_mode actually targeted --
        a 'first' pick vs its first-PA result, an 'all' pick vs its blended all-PA rate."""
        if e.get("pa_mode", "all") == "first":
            return 1.0 if e["actual_first_pa_over"] else 0.0
        total = e["actual_all_pa_total"] or 0
        return (e["actual_all_pa_over_count"] or 0) / total if total else 0.0

    def bucket_stats(rows, label_fn):
        buckets = defaultdict(lambda: {"n": 0, "first_pa_over": 0, "all_pa_over": 0, "all_pa_total": 0, "aligned_hits": 0.0})
        for e in rows:
            b = buckets[label_fn(e)]
            b["n"] += 1
            b["first_pa_over"] += 1 if e["actual_first_pa_over"] else 0
            b["all_pa_over"] += e["actual_all_pa_over_count"] or 0
            b["all_pa_total"] += e["actual_all_pa_total"] or 0
            b["aligned_hits"] += aligned_hit(e)
        return buckets

    print("\nBy confidence tier:")
    for conf, b in sorted(bucket_stats(resolved, lambda e: e["confidence"]).items()):
        fp_rate = b["first_pa_over"] / b["n"] * 100 if b["n"] else 0
        ap_rate = b["all_pa_over"] / b["all_pa_total"] * 100 if b["all_pa_total"] else 0
        aligned_rate = b["aligned_hits"] / b["n"] * 100 if b["n"] else 0
        print(f"  {conf:<8} n={b['n']:<4} aligned hit rate: {aligned_rate:.1f}%   "
              f"(first-PA: {b['first_pa_over']}/{b['n']} [{fp_rate:.1f}%]   "
              f"all-PA: {b['all_pa_over']}/{b['all_pa_total']} [{ap_rate:.1f}%])")

    print("\nBy PA mode:")
    for mode, b in sorted(bucket_stats(resolved, lambda e: e.get("pa_mode", "all")).items()):
        aligned_rate = b["aligned_hits"] / b["n"] * 100 if b["n"] else 0
        print(f"  {mode:<8} n={b['n']:<4} aligned hit rate: {aligned_rate:.1f}%")

    avg_pred = sum(e["predicted_probability"] for e in resolved) / len(resolved)
    total_aligned = sum(aligned_hit(e) for e in resolved)
    print(f"\nOverall: n={len(resolved)}   avg predicted P(4+)={avg_pred*100:.1f}%   "
          f"aligned hit rate={total_aligned/len(resolved)*100:.1f}% "
          f"(each pick judged against the outcome its own pa_mode targeted)")


if __name__ == "__main__":
    main()
