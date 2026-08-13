"""Field-average reference stats and the human-readable "why" bullets per batter."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .metrics import WindowStats, _rate_stats_for_pitch_subset


def field_reference(all_pitch_dfs: list[pd.DataFrame]) -> WindowStats:
    """Pooled pitch-level rates across every player pulled this run -- used as the
    'what's normal in this data pull' comparison point in the explanation bullets.
    Not a claimed official MLB league average, just this run's own sample."""
    frames = [d for d in all_pitch_dfs if d is not None and not d.empty]
    if not frames:
        return WindowStats()
    pooled = pd.concat(frames, ignore_index=True)
    if {"game_pk", "at_bat_number", "pitch_number"}.issubset(pooled.columns):
        pooled = pooled.drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"])
    from .metrics import build_pa_table
    pa = build_pa_table(pooled)
    return _rate_stats_for_pitch_subset(pooled, pa)


def pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x*100:.1f}%"


def key_stat_bullets(
    batter_windows: dict,
    batter_hand: WindowStats,
    pitcher_windows: dict,
    pitcher_hand: WindowStats,
    field_ref: WindowStats,
    batter_side: str,
    pitcher_hand_code: str,
) -> list[str]:
    """Pick the 2-3 stats with the largest deviation from field average that plausibly
    explain a long/short-count matchup, phrased as plain sentences."""
    b3 = batter_windows["3yr"]
    p3 = pitcher_windows["3yr"]

    candidates = []

    if batter_hand.o_swing_pct is not None and field_ref.o_swing_pct:
        dev = field_ref.o_swing_pct - batter_hand.o_swing_pct  # positive = more patient than field
        candidates.append((
            abs(dev),
            f"Batter O-Swing% {pct(batter_hand.o_swing_pct)} vs {pitcher_hand_code}HP "
            f"({'well below' if dev > 0.03 else 'below' if dev > 0 else 'above'} the "
            f"{pct(field_ref.o_swing_pct)} field avg) -> "
            f"{'chases less, works deeper counts' if dev > 0 else 'chases more, ends PAs early'}",
        ))

    if pitcher_hand.zone_pct is not None and field_ref.zone_pct:
        dev = pitcher_hand.zone_pct - field_ref.zone_pct  # positive = pounds zone more than field
        candidates.append((
            abs(dev),
            f"Pitcher Zone% {pct(pitcher_hand.zone_pct)} vs {batter_side}HB "
            f"({'above' if dev > 0 else 'below'} the {pct(field_ref.zone_pct)} field avg) -> "
            f"{'attacks the zone, shorter counts likely' if dev > 0 else 'nibbles more, longer counts likely'}",
        ))

    if pitcher_hand.fps_pct is not None and field_ref.fps_pct:
        dev = pitcher_hand.fps_pct - field_ref.fps_pct
        candidates.append((
            abs(dev),
            f"Pitcher first-pitch-strike% {pct(pitcher_hand.fps_pct)} "
            f"({'above' if dev > 0 else 'below'} the {pct(field_ref.fps_pct)} field avg) -> "
            f"{'gets ahead early, shortens PAs' if dev > 0 else 'falls behind more, PAs run longer'}",
        ))

    if batter_hand.contact_pct is not None and field_ref.contact_pct:
        dev = field_ref.contact_pct - batter_hand.contact_pct  # positive = whiffs more than field
        candidates.append((
            abs(dev) * 0.7,  # slightly downweighted -- ambiguous signal (can end in K OR foul off 2-strike)
            f"Batter contact% {pct(batter_hand.contact_pct)} "
            f"({'below' if dev > 0 else 'above'} the {pct(field_ref.contact_pct)} field avg)",
        ))

    if b3.four_plus_rate is not None:
        candidates.append((
            0.35,
            f"Batter's own {pct(b3.four_plus_rate)} rate of 4+-pitch PAs over the last 3 seasons "
            f"(n={b3.n_pa} PA)",
        ))

    if p3.four_plus_rate is not None:
        candidates.append((
            0.35,
            f"Pitcher allows 4+ pitches in {pct(p3.four_plus_rate)} of PAs over the last 3 seasons "
            f"(n={p3.n_pa} PA)",
        ))

    candidates.sort(key=lambda c: -c[0])
    return [text for _, text in candidates[:3]]
