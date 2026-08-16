#!/usr/bin/env python
"""
Predicts which batters in tonight's lineups are most likely to see 4+ pitches
in a plate appearance against the probable starter, for the "PA goes over 3.5
pitches" prop market.

Usage:
    python predict_game.py --team1 TOR --team2 BOS --date 2026-08-11 \
        --pitcher1 "Dylan Cease" --pitcher2 "Patrick Sandoval"

--team1/--pitcher1 and --team2/--pitcher2 are paired: pitcher1 is assumed to be
starting FOR team1 (and therefore faces team2's lineup), pitcher2 for team2.

See README.md for the data sources, model design, and known limitations.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Windows consoles default stdout to cp1252, which can't encode accented names
# or arrows in our output -- force UTF-8 so the report never crashes mid-print.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from mlbtool import statsapi_client as sc
from mlbtool import statcast_data as sd
from mlbtool import metrics as mx
from mlbtool import model as mdl
from mlbtool import formatting as fmt
from mlbtool import results_log


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--team1", required=True, help="Team 1 abbreviation, e.g. TOR")
    p.add_argument("--team2", required=True, help="Team 2 abbreviation, e.g. BOS")
    p.add_argument("--date", required=True, help="Game date, YYYY-MM-DD")
    p.add_argument("--pitcher1", required=True, help="Probable starter for team1")
    p.add_argument("--pitcher2", required=True, help="Probable starter for team2")
    p.add_argument("--workers", type=int, default=6, help="Parallel data-pull threads (default 6)")
    p.add_argument("--no-log", action="store_true",
                    help="Don't append recommended picks to results/predictions_log.jsonl")
    p.add_argument("--pa-slot", choices=list(mx.PA_SLOTS), default="1",
                    help="Which times-through-the-order slot to condition on: '1' (default) = "
                         "batter's 1st PA of the game / pitcher's 1st time through the order. "
                         "'2' / '3' = that specific trip. '4+' = 4th trip or later (bucketed for "
                         "sample size). 'all' = blend every PA together (the tool's original "
                         "behavior). Match this to whichever specific PA your market is actually "
                         "pricing -- a live market on a batter's 2nd at-bat needs '2', not the "
                         "default '1'.")
    return p.parse_args()


def load_player_profile(player_id: int, role: str, as_of_date: str, current_season: int, pa_slot: str):
    raw_df = sd.pull_pitcher(player_id, as_of_date) if role == "pitcher" else sd.pull_batter(player_id, as_of_date)
    raw_pa = mx.build_pa_table(raw_df)
    working_df, _ = mx.filter_by_pa_slot(raw_df, raw_pa, role, pa_slot)
    windows, pa, eligible_pks = mx.get_all_windows(working_df, role, current_season)
    start_n = len(eligible_pks) if role == "pitcher" else None
    return {
        "pitch_df": working_df,  # pa-slot-filtered: used for windows + hand splits, kept consistent
        "raw_pitch_df": raw_df,  # always unfiltered: used for BvP, which needs every scrap of sample
        "pa": pa,
        "windows": windows,
        "start_n": start_n,
    }


def main():
    args = parse_args()
    current_season = int(args.date[:4])
    t_start = time.time()

    print(f"Resolving game: {args.team1} vs {args.team2} on {args.date} ...")
    abbr_to_id = sc.team_abbr_to_id()
    for abbr in (args.team1.upper(), args.team2.upper()):
        if abbr not in abbr_to_id:
            sys.exit(f"Unknown team abbreviation '{abbr}'. Known: {sorted(abbr_to_id)}")

    game = sc.find_game(args.date, args.team1, args.team2)
    team1_id = abbr_to_id[args.team1.upper()]
    team2_id = abbr_to_id[args.team2.upper()]
    team1_side = "home" if team1_id == game.home_team_id else "away"
    team2_side = "home" if team2_id == game.home_team_id else "away"

    print(f"  Found gamePk {game.game_pk}: {game.away_abbr} @ {game.home_abbr} ({game.game_date_utc})")

    print(f"Resolving pitchers: {args.pitcher1} (team1) / {args.pitcher2} (team2) ...")
    pitcher1 = sc.lookup_player(args.pitcher1)
    pitcher2 = sc.lookup_player(args.pitcher2)
    print(f"  {pitcher1.name} throws {pitcher1.throws} | {pitcher2.name} throws {pitcher2.throws}")

    print("Checking both pitchers actually pitch like starters (not openers/bulk arms) ...")
    for p in (pitcher1, pitcher2):
        role = sc.check_pitcher_role(p, args.date)
        if role.warning:
            print(f"  ⚠️  WARNING: {role.warning}")
        elif role.recent_total:
            print(f"  {p.name}: started {role.recent_starts}/{role.recent_total} of his last "
                  f"{role.recent_total} appearances -- looks like a normal starter.")

    print("Fetching lineups (confirmed if posted, else most recent lineup used) ...")
    lineup1, conf1, fallback1_date = sc.get_lineup(game.game_pk, team1_id, team1_side, args.date)
    lineup2, conf2, fallback2_date = sc.get_lineup(game.game_pk, team2_id, team2_side, args.date)

    def describe_lineup_source(team_abbr, confirmed, fallback_date):
        if confirmed:
            return f"  {team_abbr}: CONFIRMED lineup"
        return f"  {team_abbr}: lineup NOT YET CONFIRMED -- using last used lineup (from {fallback_date}), projected"

    print(describe_lineup_source(args.team1, conf1, fallback1_date))
    print(describe_lineup_source(args.team2, conf2, fallback2_date))

    # ---- pull all statcast data in parallel (with disk caching baked in) ----
    print("Pulling Statcast histories (cached same-day; first run of the day may take a few minutes) ...")
    jobs = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        jobs[ex.submit(load_player_profile, pitcher1.id, "pitcher", args.date, current_season, args.pa_slot)] = ("pitcher", pitcher1.id)
        jobs[ex.submit(load_player_profile, pitcher2.id, "pitcher", args.date, current_season, args.pa_slot)] = ("pitcher", pitcher2.id)
        for slot in lineup1 + lineup2:
            jobs[ex.submit(load_player_profile, slot.player.id, "batter", args.date, current_season, args.pa_slot)] = ("batter", slot.player.id)

        profiles = {}
        for fut in as_completed(jobs):
            role, pid = jobs[fut]
            try:
                profiles[(role, pid)] = fut.result()
            except Exception as e:
                print(f"  WARNING: failed to pull {role} {pid}: {e}")
                profiles[(role, pid)] = None

    def prof(role, pid):
        return profiles.get((role, pid))

    # ---- league / field baselines ----
    all_pa_tables = []
    all_pitch_dfs = []
    for (role, pid), p in profiles.items():
        if p is not None:
            all_pa_tables.append(p["pa"])
            all_pitch_dfs.append(p["pitch_df"])
    league_p0 = mdl.league_baseline(all_pa_tables)
    field_ref = fmt.field_reference(all_pitch_dfs)
    print(f"  Field baseline (this pull, {len(mdl.dedupe_pa(all_pa_tables))} unique PA): "
          f"{league_p0*100:.1f}% of PAs go 4+ pitches")

    hand_baseline_cache = {}

    def get_hand_baseline(batter_side, pitcher_hand):
        key = (batter_side, pitcher_hand)
        if key not in hand_baseline_cache:
            hand_baseline_cache[key] = mdl.hand_baseline(all_pa_tables, batter_side, pitcher_hand, league_p0)
        return hand_baseline_cache[key]

    # ---- evaluate every batter vs the opposing starter ----
    results = []
    matchups = [
        (lineup1, args.team1.upper(), pitcher2, args.team2.upper()),
        (lineup2, args.team2.upper(), pitcher1, args.team1.upper()),
    ]
    for lineup, team_abbr, pitcher, opp_abbr in matchups:
        pp = prof("pitcher", pitcher.id)
        if pp is None:
            print(f"  Skipping {team_abbr} lineup -- no pitcher data for {pitcher.name}")
            continue
        for slot in lineup:
            bp = prof("batter", slot.player.id)
            if bp is None:
                continue
            batter_side = slot.player.bats or "R"
            pitcher_hand_split = mx.get_hand_split(pp["pitch_df"], pp["pa"], "pitcher", batter_side)
            batter_hand_split = mx.get_hand_split(bp["pitch_df"], bp["pa"], "batter", pitcher.throws or "R")
            # BvP always draws from the unfiltered history -- these samples are already tiny
            # (often 0-15 PA), and further restricting to "first PA of the game against this
            # exact pitcher" would leave almost nothing to work with.
            bvp = mx.get_bvp(bp["raw_pitch_df"], pitcher.id)
            hand_base = get_hand_baseline(batter_side, pitcher.throws or "R")

            res = mdl.evaluate_matchup(
                batter_windows=bp["windows"],
                batter_hand_split=batter_hand_split,
                batter_name=slot.player.name,
                batter_id=slot.player.id,
                order=slot.order,
                team_abbr=team_abbr,
                pitcher_windows=pp["windows"],
                pitcher_hand_split=pitcher_hand_split,
                pitcher_start_n=pp["start_n"] or 0,
                pitcher_name=pitcher.name,
                opp_pitcher_abbr=opp_abbr,
                league_p0=league_p0,
                field_hand_baseline=hand_base,
                bvp=bvp,
            )
            bullets = fmt.key_stat_bullets(
                bp["windows"], batter_hand_split, pp["windows"], pitcher_hand_split,
                field_ref, batter_side, pitcher.throws or "R",
            )
            results.append((res, bullets))

    results.sort(key=lambda r: -r[0].probability)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.0f}s.\n")
    print_report(args, game, results, league_p0)


def print_report(args, game, results, league_p0):
    W = 100
    print("=" * W)
    print(f" {args.team1.upper()} @ {args.team2.upper()} / {args.pitcher1} vs {args.pitcher2}  --  {args.date}")
    print(f" Market: PA over 3.5 pitches @ 1.6x  |  Breakeven implied probability: {mdl.BREAKEVEN*100:.1f}%")
    slot_notes = {
        "1": "batter's 1st PA of game / pitcher's 1st time through order",
        "2": "batter's 2nd PA of game / pitcher's 2nd time through order",
        "3": "batter's 3rd PA of game / pitcher's 3rd time through order",
        "4+": "batter's 4th+ PA of game / pitcher's 4th+ time through order",
        "all": "all PA blended together",
    }
    print(f" PA slot: {args.pa_slot} ({slot_notes.get(args.pa_slot, '')})")
    print("=" * W)

    print("\nFULL RANKING (all batters, both lineups, vs the opposing starter)\n")
    print(" '*' = clears the meaningful-edge bar below (a real recommendation, not just a rank)")
    header = f"{'#':<3}{'Batter':<24}{'Tm':<5}{'Ord':<5}{'vs':<20}{'P(4+)':>8}{'Edge':>8}{'Conf':>8}  "
    print(header)
    print("-" * len(header))
    for i, (r, _) in enumerate(results, 1):
        edge_str = f"{r.edge*100:+.1f}%"
        mark = "*" if mdl.clears_meaningful_edge(r.edge, r.confidence) else ""
        print(f"{i:<3}{r.batter_name:<24}{r.team_abbr:<5}{ordinal(r.order):<5}{r.pitcher_name:<20}{r.probability*100:>7.1f}%{edge_str:>8}{r.confidence:>8}  {mark}")

    print("\n" + "=" * W)
    print(" RECOMMENDED -- every pick that clears a real edge (no cap, no coin-flip padding)")
    print(f" Bar: edge >= +3.0pts (High conf) / +5.0pts (Medium) / +8.0pts (Low) -- thin-sample")
    print(f" picks need a bigger observed edge before they're distinguishable from a coin flip.")
    print("=" * W)

    qualifying = [(r, b) for r, b in results if mdl.clears_meaningful_edge(r.edge, r.confidence)]

    if not getattr(args, "no_log", False):
        n_logged = results_log.log_picks(
            game.game_pk, args.date, args.team1.upper(), args.team2.upper(), qualifying, pa_slot=args.pa_slot
        )
        if n_logged:
            print(f"\n(logged {n_logged} pick(s) to results/predictions_log.jsonl for later resolution)")

    if not qualifying:
        print(f"\nNo batter in this game clears a meaningful edge over the {mdl.BREAKEVEN*100:.1f}% breakeven.")
        best = results[0][0] if results else None
        if best is not None:
            print(f"Closest was {best.batter_name} at {best.probability*100:.1f}% ({best.edge*100:+.1f} pts, "
                  f"{best.confidence} confidence) -- not enough margin to call it a real edge.")
        print("Treat this game as a pass on the 'over 3.5 pitches' side.")
        print()
        return

    for i, (r, bullets) in enumerate(qualifying, 1):
        print(f"\n{i}. {r.batter_name} ({r.team_abbr}) vs {r.pitcher_name}  --  batting {ordinal(r.order)}")
        print(f"   Estimated P(4+ pitches): {r.probability*100:.1f}%   "
              f"Breakeven needed: {mdl.BREAKEVEN*100:.1f}%   "
              f"Edge: {r.edge*100:+.1f} pts   Confidence: {r.confidence}")
        for b in bullets:
            print(f"     - {b}")
        if r.bvp_n > 0:
            print(f"     - BvP history: {r.bvp_n} career PA vs {r.pitcher_name}, "
                  f"{fmt.pct(r.bvp_rate)} went 4+ (small sample, lightly weighted into the estimate above)")
        else:
            print(f"     - BvP history: no meaningful head-to-head sample (0 PA)")

    print()
    print(f"{len(qualifying)} pick(s) in this game clear a real edge over the {mdl.BREAKEVEN*100:.1f}% breakeven "
          f"at a meaningful margin for their confidence level -- every one of them, not just the top 3.")
    print()


def ordinal(n: int) -> str:
    return {1: "1st", 2: "2nd", 3: "3rd"}.get(n, f"{n}th")


if __name__ == "__main__":
    main()
