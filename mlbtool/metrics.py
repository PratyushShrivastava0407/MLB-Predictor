"""
Turns a raw Statcast pitch-level DataFrame (one row per pitch) into plate-appearance
level facts and rate stats, sliced into the windows the user asked for:
last-15-games, last-30-games, season-to-date, and a 3-season pool (see
statcast_data.py for why "3yr" stands in for "career").

Nothing in here does any shrinkage/weighting -- that happens in model.py once we
know the league baseline (which itself is pooled across every player in a given
run). This module just computes plain empirical rates per slice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

SWING_DESC = {
    "swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
    "hit_into_play", "foul_bunt", "missed_bunt",
}
WHIFF_DESC = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
FOUL_DESC = {"foul", "foul_tip", "foul_bunt"}
BB_EVENTS = {"walk", "intent_walk"}
K_EVENTS = {"strikeout", "strikeout_double_play"}


@dataclass
class WindowStats:
    n_pa: int = 0
    n_pitches: int = 0
    four_plus_rate: Optional[float] = None
    avg_pitches: Optional[float] = None
    zone_pct: Optional[float] = None
    o_swing_pct: Optional[float] = None
    z_swing_pct: Optional[float] = None
    contact_pct: Optional[float] = None
    whiff_pct: Optional[float] = None
    fps_pct: Optional[float] = None
    foul_rate: Optional[float] = None
    bb_pct: Optional[float] = None
    k_pct: Optional[float] = None
    top_pitches: list = field(default_factory=list)


def build_pa_table(pitch_df: pd.DataFrame) -> pd.DataFrame:
    if pitch_df is None or pitch_df.empty:
        return pd.DataFrame(
            columns=["game_pk", "at_bat_number", "game_date", "game_year",
                     "batter", "pitcher", "stand", "p_throws", "n_pitches",
                     "four_plus", "bb", "k"]
        )
    df = pitch_df.copy()
    df["events"] = df["events"].where(df["events"].notna(), None)

    def last_event(s: pd.Series):
        s = s.dropna()
        return s.iloc[-1] if len(s) else None

    agg_spec = dict(
        game_date=("game_date", "first"),
        game_year=("game_year", "first"),
        batter=("batter", "first"),
        pitcher=("pitcher", "first"),
        stand=("stand", "first"),
        p_throws=("p_throws", "first"),
        n_pitches=("pitch_number", "max"),
        events=("events", last_event),
    )
    # Times-through-the-order info, when Statcast provides it (it does for any pull
    # from a reasonably recent season) -- used to isolate "batter's first PA of the
    # game" / "pitcher's first time through the lineup" from later, different-regime
    # PAs. Optional so a pull missing these columns degrades gracefully instead of
    # crashing.
    if "n_priorpa_thisgame_player_at_bat" in df.columns:
        agg_spec["prior_pa_this_game"] = ("n_priorpa_thisgame_player_at_bat", "first")
    if "n_thruorder_pitcher" in df.columns:
        agg_spec["times_thru_order"] = ("n_thruorder_pitcher", "first")

    pa = df.groupby(["game_pk", "at_bat_number"]).agg(**agg_spec).reset_index()
    pa["four_plus"] = pa["n_pitches"] >= 4
    pa["bb"] = pa["events"].isin(BB_EVENTS)
    pa["k"] = pa["events"].isin(K_EVENTS)
    pa["game_date"] = pd.to_datetime(pa["game_date"])
    if "prior_pa_this_game" in pa.columns:
        pa["pa_num_this_game"] = pa["prior_pa_this_game"] + 1  # 1-indexed: 1 = first PA of the game
    return pa


PA_SLOTS = ("1", "2", "3", "4+", "all")


def filter_by_pa_slot(pitch_df: pd.DataFrame, pa: pd.DataFrame, role: str, pa_slot: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Restrict to a specific times-through-the-order slot: for a batter, their Nth
    PA of the game; for a pitcher, his Nth time through the lineup. These aren't
    quite the same slice (a pitcher's "time through the order" window covers
    ~3-4 batters, a batter's "Nth PA" window is exactly 1 PA/game), but each
    isolates the same real effect from its own side of the matchup -- performance
    in the 1st, 2nd, 3rd meeting of a game differs (the well-documented "times
    through the order" effect), and a live market prices whichever specific PA
    is coming up next, not some blend across the whole game.

    pa_slot: '1', '2', '3', '4+' (4th or later, bucketed together since sample
    size gets thin one at a time beyond the 3rd trip), or 'all' (no filtering --
    the tool's original behavior, blending every PA together).
    """
    if pa_slot == "all" or pa.empty:
        return pitch_df, pa

    col = "pa_num_this_game" if role == "batter" else "times_thru_order"
    if col not in pa.columns:
        return pitch_df, pa  # column unavailable (older/incomplete pull) -- fall back to unfiltered

    if pa_slot == "4+":
        keep = pa[pa[col] >= 4]
    elif pa_slot in ("1", "2", "3"):
        keep = pa[pa[col] == int(pa_slot)]
    else:
        raise ValueError(f"Unknown pa_slot '{pa_slot}', expected one of {PA_SLOTS}")

    keep_keys = set(zip(keep["game_pk"], keep["at_bat_number"]))
    pitch_index = pd.MultiIndex.from_arrays([pitch_df["game_pk"], pitch_df["at_bat_number"]])
    pitch_keep = pitch_df[pitch_index.isin(keep_keys)]
    return pitch_keep, keep.reset_index(drop=True)


def determine_start_game_pks(pitch_df: pd.DataFrame, pa: pd.DataFrame) -> set:
    """Games where this pitcher faced the leadoff batter (at_bat_number==1) --
    a simple, robust-enough proxy for 'this was a start' vs a relief outing."""
    if pa.empty:
        return set()
    starts = pa.groupby("game_pk")["at_bat_number"].min()
    return set(starts[starts == 1].index)


def _rate_stats_for_pitch_subset(pitch_sub: pd.DataFrame, pa_sub: pd.DataFrame) -> WindowStats:
    ws = WindowStats()
    ws.n_pa = len(pa_sub)
    ws.n_pitches = len(pitch_sub)
    if ws.n_pa == 0:
        return ws

    ws.four_plus_rate = float(pa_sub["four_plus"].mean())
    ws.avg_pitches = float(pa_sub["n_pitches"].mean())
    ws.bb_pct = float(pa_sub["bb"].mean())
    ws.k_pct = float(pa_sub["k"].mean())

    if ws.n_pitches > 0:
        zone = pd.to_numeric(pitch_sub["zone"], errors="coerce")
        in_zone = zone.between(1, 9)
        out_zone = zone.between(11, 14)
        swing = pitch_sub["description"].isin(SWING_DESC)
        whiff = pitch_sub["description"].isin(WHIFF_DESC)
        foul = pitch_sub["description"].isin(FOUL_DESC)

        ws.zone_pct = float(in_zone.mean())
        ws.whiff_pct = float(whiff.sum() / swing.sum()) if swing.sum() > 0 else None
        ws.contact_pct = (
            float((swing & ~whiff).sum() / swing.sum()) if swing.sum() > 0 else None
        )
        ws.foul_rate = float(foul.mean())

        if out_zone.sum() > 0:
            ws.o_swing_pct = float((swing & out_zone).sum() / out_zone.sum())
        if in_zone.sum() > 0:
            ws.z_swing_pct = float((swing & in_zone).sum() / in_zone.sum())

        first_pitches = pitch_sub[pitch_sub["pitch_number"] == 1]
        if len(first_pitches) > 0:
            ws.fps_pct = float((first_pitches["type"] != "B").mean())

        if "pitch_name" in pitch_sub.columns:
            vc = pitch_sub["pitch_name"].dropna().value_counts(normalize=True).head(3)
            ws.top_pitches = [(name, round(pct * 100, 1)) for name, pct in vc.items()]

    return ws


def compute_window(pitch_df: pd.DataFrame, pa: pd.DataFrame, game_pks: Optional[set]) -> WindowStats:
    if game_pks is None:
        pitch_sub, pa_sub = pitch_df, pa
    else:
        pitch_sub = pitch_df[pitch_df["game_pk"].isin(game_pks)]
        pa_sub = pa[pa["game_pk"].isin(game_pks)]
    return _rate_stats_for_pitch_subset(pitch_sub, pa_sub)


def get_all_windows(pitch_df: pd.DataFrame, role: str, current_season: int) -> dict:
    """role: 'pitcher' restricts L15/L30/season windows to *starts* only."""
    pa = build_pa_table(pitch_df)
    if pa.empty:
        empty = WindowStats()
        return {"L15": empty, "L30": empty, "season": empty, "3yr": empty}, pa, set()

    if role == "pitcher":
        eligible_pks = determine_start_game_pks(pitch_df, pa)
    else:
        eligible_pks = set(pa["game_pk"].unique())

    pa_elig = pa[pa["game_pk"].isin(eligible_pks)]
    game_dates = (
        pa_elig[["game_pk", "game_date"]].drop_duplicates().sort_values("game_date", ascending=False)
    )
    l15_pks = set(game_dates.head(15)["game_pk"])
    l30_pks = set(game_dates.head(30)["game_pk"])
    season_pks = set(pa_elig[pa_elig["game_year"] == current_season]["game_pk"])

    windows = {
        "L15": compute_window(pitch_df, pa, l15_pks),
        "L30": compute_window(pitch_df, pa, l30_pks),
        "season": compute_window(pitch_df, pa, season_pks),
        "3yr": compute_window(pitch_df, pa, eligible_pks),
    }
    return windows, pa, eligible_pks


def get_hand_split(pitch_df: pd.DataFrame, pa: pd.DataFrame, role: str, hand: str) -> WindowStats:
    """role='pitcher' -> split by batter stand; role='batter' -> split by pitcher throws."""
    if pitch_df.empty or not hand:
        return WindowStats()
    col = "stand" if role == "pitcher" else "p_throws"
    pitch_sub = pitch_df[pitch_df[col] == hand]
    pa_sub = pa[pa[col] == hand] if col in pa.columns else pa.iloc[0:0]
    return _rate_stats_for_pitch_subset(pitch_sub, pa_sub)


def get_bvp(batter_pitch_df: pd.DataFrame, pitcher_id: int) -> WindowStats:
    """Batter-vs-this-pitcher history, pulled from the batter's own 3yr log."""
    if batter_pitch_df.empty:
        return WindowStats()
    sub = batter_pitch_df[batter_pitch_df["pitcher"] == pitcher_id]
    if sub.empty:
        return WindowStats()
    pa = build_pa_table(sub)
    return _rate_stats_for_pitch_subset(sub, pa)
