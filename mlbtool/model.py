"""
Scoring model: turns each batter's window stats + the opposing pitcher's window
stats into P(this PA goes 4+ pitches).

Design, in order:

1. Shrinkage cascade per player. Instead of hand-picking one window (L15? season?)
   we shrink each window's raw rate toward the *next larger, more stable* window
   using a simple empirical-Bayes pseudo-count: p_shrunk = (n*p_raw + k*prior) /
   (n+k). L15 shrinks toward L30, L30 toward season, season toward the 3yr pool,
   and the 3yr pool shrinks toward the league baseline. This means a batter's
   hot/cold streak in his last 15 games can't swing the estimate wildly if 15 PA
   is a small sample -- but it isn't thrown out either, just weighted by how much
   signal it actually carries.
2. The four shrunk windows are then blended with fixed recency-tilted weights
   (L15 gets the most weight, 3yr the least) into one "overall rate" per player.
3. That overall rate is itself shrunk toward the handedness-specific rate (how
   this batter does vs this pitcher's throwing hand / how this pitcher does vs
   this batter's side) using the same pseudo-count logic -- so a small
   platoon-split sample doesn't overwhelm the stable overall number.
4. Batter and pitcher are combined in log-odds space, NOT averaged:
       logit(matchup) = logit(batter_hand_rate) + logit(pitcher_hand_rate) - logit(field_hand_baseline)
   This is the standard way to combine two independent deviations from a shared
   baseline (equivalent to a 2-feature naive-Bayes log-odds combination). It
   means a patient batter AND a pitcher who nibbles both push the estimate up;
   a patient batter facing a pitcher who pounds the zone still gets pulled back
   toward average by the pitcher's term, rather than just splitting the
   difference the way a naive (batter+pitcher)/2 average would.
5. Batter-vs-pitcher head-to-head history, if any, is folded in last with a small
   pseudo-count (kappa=15 PA) so even a handful of BvP PAs barely move the number
   -- shown separately in the output so you can see it, not hidden inside a
   blended figure.

No park factor / day-night adjustment: pitch-count-per-PA has only a weak,
inconsistent relationship with park/time-of-day in the literature relative to
batter/pitcher approach, and we don't have the PA-level sample size in a single
run to estimate it without just fitting noise. Left out on purpose.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .metrics import WindowStats

EPS = 1e-6

KAPPA_3YR = 40
KAPPA_SEASON = 25
KAPPA_L30 = 20
KAPPA_L15 = 15
KAPPA_HAND = 30
KAPPA_BVP = 15

BLEND_WEIGHTS = {"L15": 0.40, "L30": 0.25, "season": 0.20, "3yr": 0.15}

BREAKEVEN = 0.625  # implied by 1.6x price


def logit(p: float) -> float:
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def inv_logit(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def shrink(n: int, p_raw: Optional[float], prior: float, kappa: float) -> float:
    if not n or p_raw is None:
        return prior
    return (n * p_raw + kappa * prior) / (n + kappa)


@dataclass
class PlayerCascade:
    p_3yr: float
    p_season: float
    p_L30: float
    p_L15: float
    overall: float
    n_3yr: int
    n_season: int
    n_L30: int
    n_L15: int


def cascade(windows: dict, league_p0: float) -> PlayerCascade:
    w3 = windows["3yr"]
    p_3yr = shrink(w3.n_pa, w3.four_plus_rate, league_p0, KAPPA_3YR)

    ws = windows["season"]
    p_season = shrink(ws.n_pa, ws.four_plus_rate, p_3yr, KAPPA_SEASON)

    w30 = windows["L30"]
    p_L30 = shrink(w30.n_pa, w30.four_plus_rate, p_season, KAPPA_L30)

    w15 = windows["L15"]
    p_L15 = shrink(w15.n_pa, w15.four_plus_rate, p_L30, KAPPA_L15)

    overall = (
        BLEND_WEIGHTS["L15"] * p_L15
        + BLEND_WEIGHTS["L30"] * p_L30
        + BLEND_WEIGHTS["season"] * p_season
        + BLEND_WEIGHTS["3yr"] * p_3yr
    )
    return PlayerCascade(
        p_3yr=p_3yr, p_season=p_season, p_L30=p_L30, p_L15=p_L15, overall=overall,
        n_3yr=w3.n_pa, n_season=ws.n_pa, n_L30=w30.n_pa, n_L15=w15.n_pa,
    )


def hand_adjusted(overall: float, hand_stats: WindowStats, kappa: float = KAPPA_HAND) -> tuple:
    n = hand_stats.n_pa or 0
    p = shrink(n, hand_stats.four_plus_rate, overall, kappa)
    return p, n


def dedupe_pa(pa_tables: list[pd.DataFrame]) -> pd.DataFrame:
    if not pa_tables:
        return pd.DataFrame(columns=["game_pk", "at_bat_number", "four_plus"])
    pooled = pd.concat([t for t in pa_tables if not t.empty], ignore_index=True)
    if pooled.empty:
        return pooled
    return pooled.drop_duplicates(subset=["game_pk", "at_bat_number"])


def league_baseline(pa_tables: list[pd.DataFrame]) -> float:
    pooled = dedupe_pa(pa_tables)
    if pooled.empty or pooled["four_plus"].isna().all():
        return 0.46  # rough modern-MLB fallback if a run somehow pulls no data
    return float(pooled["four_plus"].mean())


def hand_baseline(pa_tables: list[pd.DataFrame], batter_hand: str, pitcher_hand: str, fallback: float) -> float:
    pooled = dedupe_pa(pa_tables)
    if pooled.empty or "stand" not in pooled.columns:
        return fallback
    sub = pooled[(pooled["stand"] == batter_hand) & (pooled["p_throws"] == pitcher_hand)]
    if len(sub) < 30:
        return fallback
    return float(sub["four_plus"].mean())


def confidence_label(batter_hand_n: int, pitcher_start_n: int) -> str:
    if pitcher_start_n < 6 or batter_hand_n < 25:
        return "Low"
    if pitcher_start_n < 15 or batter_hand_n < 60:
        return "Medium"
    return "High"


@dataclass
class MatchupResult:
    batter_name: str
    batter_id: int
    pitcher_name: str
    order: int
    team_abbr: str
    opp_pitcher_abbr: str
    probability: float
    matchup_p_pre_bvp: float
    batter_hand_p: float
    pitcher_hand_p: float
    batter_hand_n: int
    pitcher_hand_n: int
    pitcher_start_n: int
    bvp_n: int
    bvp_rate: Optional[float]
    confidence: str
    field_hand_baseline: float
    stat_candidates: list = field(default_factory=list)  # (label, value_str, note)

    @property
    def edge(self) -> float:
        return self.probability - BREAKEVEN


def evaluate_matchup(
    *,
    batter_windows: dict,
    batter_hand_split: WindowStats,
    batter_name: str,
    batter_id: int,
    order: int,
    team_abbr: str,
    pitcher_windows: dict,
    pitcher_hand_split: WindowStats,
    pitcher_start_n: int,
    pitcher_name: str,
    opp_pitcher_abbr: str,
    league_p0: float,
    field_hand_baseline: float,
    bvp: WindowStats,
) -> MatchupResult:
    batter_cascade = cascade(batter_windows, league_p0)
    pitcher_cascade = cascade(pitcher_windows, league_p0)

    batter_hand_p, batter_hand_n = hand_adjusted(batter_cascade.overall, batter_hand_split)
    pitcher_hand_p, pitcher_hand_n = hand_adjusted(pitcher_cascade.overall, pitcher_hand_split)

    matchup_logit = logit(batter_hand_p) + logit(pitcher_hand_p) - logit(field_hand_baseline)
    matchup_p = inv_logit(matchup_logit)

    bvp_n = bvp.n_pa or 0
    final_p = shrink(bvp_n, bvp.four_plus_rate, matchup_p, KAPPA_BVP)

    conf = confidence_label(batter_hand_n, pitcher_start_n)

    return MatchupResult(
        batter_name=batter_name,
        batter_id=batter_id,
        pitcher_name=pitcher_name,
        order=order,
        team_abbr=team_abbr,
        opp_pitcher_abbr=opp_pitcher_abbr,
        probability=final_p,
        matchup_p_pre_bvp=matchup_p,
        batter_hand_p=batter_hand_p,
        pitcher_hand_p=pitcher_hand_p,
        batter_hand_n=batter_hand_n,
        pitcher_hand_n=pitcher_hand_n,
        pitcher_start_n=pitcher_start_n,
        bvp_n=bvp_n,
        bvp_rate=bvp.four_plus_rate,
        confidence=conf,
        field_hand_baseline=field_hand_baseline,
    )
