# NFL Matchup Scout

Profiles every team's offense and defense across situational splits, flags statistical outliers vs
league average as strengths/weaknesses (z-scores per situation), then cross-references two teams to
find where one side's strength meets the other's weakness, ranked by magnitude × expected frequency.

```
HIGH-LEVERAGE EXPLOIT — CHI offense: Off-tackle run vs NYG defense. CHI averages +0.13 EPA/play
with a 50% success rate here (z +2.0, n=117), while NYG allows +0.26 EPA/play (z -2.4, n=96).
Occurs about 6.9 times per game for CHI historically (~7.7 projected vs NYG) ...
```

## Run

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/streamlit run app.py              # UI
.venv/bin/python -m nfl_scout KC LV --seasons 2024 2025 > report.md   # Markdown report
.venv/bin/python -m pytest -q               # tests (synthetic data, no network)
```

The first run downloads nflverse release parquet files into `data/cache/` (~25 MB per season).
The current season's files refresh every 24 h.

## Data
| Source | Used for |
|---|---|
| `nflverse-data` → `pbp` (built by `nflverse-pbp`) | EPA, success, down/distance, field position, run gap/location, dropback |
| `nflverse-data` → `pbp_participation` | offensive/defensive personnel, formation, box count, pass rushers, pressure, man/zone |
| `nflverse-data` → `ftn_charting` | play action |
| `ngs-data` (`nextgen_stats` release) | team context: time to throw, aggressiveness, CPOE, RYOE, 8-man-box rate |

Participation lags play-by-play. When a season has none (e.g. the in-progress 2026 season),
the personnel, pressure, coverage and box splits drop out for that season and the app says so.

## Layout
- `nfl_scout/config.py`: data URLs, dimensions, the 21 split families, default thresholds
- `nfl_scout/features.py`: filters to scrimmage plays and tags every dimension
- `nfl_scout/profiles.py`: per-team situational EPA/SR, stabilization, z-scores, flags
- `nfl_scout/matchup.py`: cross-reference, ranking, nested-finding folding, report text
- `nfl_scout/ngs.py`: Next Gen Stats team context
- `app.py`: Streamlit UI (matchup exploits, team profile, league heatmap, methodology)

The full method is described in the app's Methodology tab and in the docstrings of `profiles.py` and `matchup.py`.
