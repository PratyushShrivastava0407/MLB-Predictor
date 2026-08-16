# MLB "PA over 3.5 pitches" predictor

Ranks a game's batters by estimated probability that a given plate appearance
against the probable starter goes 4+ pitches, for the sportsbook "over 3.5
pitches" prop.

## Usage

```
python predict_game.py --team1 TOR --team2 BOS --date 2026-08-11 \
    --pitcher1 "Dylan Cease" --pitcher2 "Patrick Sandoval"
```

`--pitcher1` is assumed to start **for** `--team1** (and therefore faces
`--team2`'s lineup); `--pitcher2` starts for `--team2`. Team abbreviations are
standard MLB codes (see error message for the full list if unsure).

First run of the day pulls ~3 seasons of Statcast pitch logs for ~20 players
(takes 20-60s with 6 parallel threads). Every rerun that day reads from the
local `cache/` folder instead — deleting `cache/` forces a fresh pull.

## What it does

1. **Game & lineups** — looks up the game via MLB Stats API's schedule
   endpoint, then the confirmed starting lineup if MLB has posted it yet
   (usually 1-3 hours before first pitch). If not posted, falls back to the
   lineup from that team's most recent completed game and labels it
   *projected, not confirmed* in the output — always check the real lineup
   before betting since bench/rest days do happen.
1.5. **Opener/bulk-arm check** — before pulling any Statcast data, each
   probable pitcher's last 5 appearances are checked for whether he actually
   faced the leadoff batter for his team that day (checking both halves of
   the 1st inning, since the home team's pitcher opens the top and the away
   team's opens the bottom). If fewer than half of his recent outings were
   true starts, the report prints a loud warning -- this pitcher is likely
   a bulk/opener-follower arm, and the lineup slots he's matched against may
   not reflect who those batters actually face. This exact pattern (a team's
   "probable starter" actually throwing 1 inning before a bulk reliever takes
   over) has shown up multiple times in testing and silently invalidates
   picks tied to that pitcher if not caught. See `check_pitcher_role` in
   `mlbtool/statsapi_client.py`.
2. **Statcast pull** — `pybaseball.statcast_pitcher` / `statcast_batter` for
   both probable starters and every batter in both lineups, current season +
   prior 2 seasons ("3yr" throughout, not full career — see *Design choices*
   below).
3. **Metrics** — from the raw pitch log we build a plate-appearance table
   (pitches per PA, walk, strikeout) and pitch-level rates (zone%, O-Swing%,
   Z-Swing%, contact%, whiff%, first-pitch-strike%, foul rate, pitch mix),
   each sliced into last-15-games / last-30-games / season / 3yr, and again
   split by opposing handedness.
4. **Model** — combines batter and pitcher tendencies into P(4+ pitches).
   See `mlbtool/model.py` docstring for the full reasoning; short version:
   - Each window is shrunk toward the next larger, more stable window
     (empirical-Bayes pseudo-counts), then blended with recency-tilted
     weights (L15 40%, L30 25%, season 20%, 3yr 15%).
   - That blended rate is shrunk toward the batter's/pitcher's
     handedness-specific rate.
   - Batter and pitcher are combined in **log-odds space**, not averaged:
     `logit(matchup) = logit(batter_hand_rate) + logit(pitcher_hand_rate) - logit(field_baseline)`.
     A patient batter vs. a zone-pounding pitcher gets pulled toward average
     by the pitcher's term instead of just splitting the difference.
   - Batter-vs-pitcher head-to-head history, if any, nudges the final number
     with a small pseudo-count (kappa=15 PA) — shown separately, not hidden.
5. **Output** — full ranking of every batter in both lineups against the
   opposing starter (combined into one pool, since either lineup is bettable
   in "today's game"), then **every** pick that clears a **meaningful edge**,
   scaled by confidence (`edge >= +3.0pts` High / `+5.0pts` Medium / `+8.0pts`
   Low — see `mlb.model.MIN_EDGE_BY_CONFIDENCE`). This list is not capped at
   3 and not padded to 3 -- a game can surface 0, 1, 4, or however many
   batters genuinely clear the bar. A pick at only +0.5pts over breakeven is
   statistical noise, not a recommendation, and is excluded regardless of
   where it ranks; a 4th or 5th batter with a real edge is included even
   though it isn't top-3 by raw rank. If nothing clears the bar, the report
   says so explicitly and tells you to pass on the game.

## Design choices / limitations (read before betting on this)

- **"3yr" not "career."** Full-career Statcast pulls for a 10+ year veteran
  are slow and largely irrelevant (approach/stuff drift over a decade). We
  use current + prior 2 seasons everywhere "career" was requested, and label
  it as such rather than overclaiming.
- **No park factor / day-night adjustment.** Pitches-per-PA has a weak and
  inconsistent relationship with park/time-of-day in the literature, and a
  single day's PA sample isn't enough to fit that signal without just fitting
  noise. Deliberately left out.
- **Lineup fallback is a real limitation, not a formality.** If MLB hasn't
  posted the lineup yet, the tool uses the last lineup that team ran out —
  correct rest days, minor injuries, or platoon shuffles are not accounted
  for. The output tells you plainly when this happened.
- **Starts vs. relief appearances** for pitchers are inferred by "did this
  pitcher face the game's first batter" — a reasonable proxy but not the
  official designation; openers/bulk-reliever roles could misclassify rarely.
- **Confidence flags are heuristic, not calibrated.** `Low` fires when a
  pitcher has under 6 starts or a batter has under 25 PA against that
  handedness in the pulled window — rules of thumb, not a fit statistical
  interval. Treat `Low` as "this number is closer to a guess than the others."
- **Field baseline is this run's own pooled sample** (~20-25k PA across the
  ~20 players pulled), not an externally sourced league constant — printed
  each run so you can see what it was.
- **No live/in-progress lineup changes.** If a late scratch happens after you
  run the tool, rerun it close to first pitch.
- This is a probability estimate from historical tendencies, not a
  guarantee — sample sizes even at "3yr" are in the hundreds to low
  thousands of PA per player, and matchup-specific (BvP) samples are almost
  always too small to lean on alone.

## Times-through-the-order (`--pa-slot`)

Every stat in this tool can be conditioned on a specific times-through-the-
order slot instead of blending a batter's/pitcher's whole game together.
This matters for two reasons: the well-documented times-through-the-order
effect (hitters and pitchers behave differently the more times they face
each other in one game), and -- more importantly if your book only offers
this market **live** -- a live line is priced on one *specific* upcoming PA,
not some blend across the whole game. If a batter is leading off the 4th
inning, that's probably his 2nd PA of the day, and pricing it off his
first-PA-only stats (or a game-wide blend) would answer the wrong question.

`--pa-slot`, default `'1'`:
  - `'1'` (default) -- batter's 1st PA of the game / pitcher's 1st time through the order
  - `'2'` / `'3'` -- that specific trip
  - `'4+'` -- 4th trip or later, bucketed together for sample size
  - `'all'` -- every PA blended together (the tool's original behavior)

**Match this to whatever your market is actually pricing.** For a live book,
that means knowing, at the moment you check a line, which numbered PA that
batter is on today -- the same way you'd check a live box score. Statcast
provides the columns needed for this directly
(`n_priorpa_thisgame_player_at_bat` for the batter side, `n_thruorder_pitcher`
for the pitcher side) -- see `mlbtool.metrics.filter_by_pa_slot`.

BvP history always uses the unfiltered, full history regardless of slot --
those samples are already so small (often 0-15 PA) that narrowing further
would leave nothing to work with.

The results log records which `pa_slot` produced each pick, and
`resolve_results.py` checks outcomes against the *specific* PA that matches
that slot (a batter's actual 2nd PA that game, not just his first meeting
with that pitcher) -- see `actual_slot_pitches`/`actual_slot_over` in the log
schema.

## Tracking real hit rates over time

Every `predict_game.py` run automatically appends its recommended picks to
`results/predictions_log.jsonl` (local only, not committed -- see
`.gitignore`; pass `--no-log` to skip). Once games go Final, run:

```
python resolve_results.py --summary
```

This pulls each logged game's actual play-by-play, finds the logged batter's
PA(s) against the logged pitcher specifically, and records both the outcome
of their **first** PA that game and **every** PA they had against that
pitcher (a market might resolve on either framing -- see the WSH@NYM
conversation in this project's history for why that distinction matters).
The `--summary` flag then prints hit rates broken out by confidence tier, so
you can check whether "High confidence" picks are actually hitting near
their predicted rate over enough samples to mean something -- rather than
reacting to any single night's results, good or bad.

## Project layout

```
predict_game.py          CLI entry point / orchestration / report printing
mlbtool/
  statsapi_client.py      MLB Stats API: games, lineups, player id/handedness
  statcast_data.py         cached pybaseball pulls (3 seasons per player)
  metrics.py                PA table + pitch-level rate stats, windowed
  model.py                   shrinkage cascade + log-odds combination
  formatting.py             field-reference stats + "why" bullet generation
  cache.py                   disk cache (JSON for lookups, parquet for pitch logs)
cache/                    on-disk cache (safe to delete anytime)
```
