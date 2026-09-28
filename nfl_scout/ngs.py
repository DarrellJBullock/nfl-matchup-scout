"""Next Gen Stats team context (nflverse ngs-data): how an offense operates, not just how well."""
from __future__ import annotations

import pandas as pd

from . import data

TEAM_FIX = {"LAR": "LA"}  # NGS uses LAR; nflverse pbp uses LA

PASS_METRICS = {
    "avg_time_to_throw": "Avg time to throw (s)",
    "aggressiveness": "Aggressiveness (% tight-window)",
    "avg_intended_air_yards": "Avg intended air yards",
    "completion_percentage_above_expectation": "CPOE",
}
RUSH_METRICS = {
    "rush_yards_over_expected_per_att": "Rush yds over expected / att",
    "percent_attempts_gte_eight_defenders": "% rushes vs 8+ box",
    "avg_time_to_los": "Avg time to LOS (s)",
    "efficiency": "Rush efficiency (dist. per yd gained)",
}


def _weighted(df: pd.DataFrame, metrics: dict[str, str], weight: str) -> pd.DataFrame:
    df = df.dropna(subset=[weight])
    agg = {}
    for col in metrics:
        valid = df[col].notna()
        num = (df.loc[valid, col] * df.loc[valid, weight]).groupby(df.loc[valid, "team"]).sum()
        den = df.loc[valid, weight].groupby(df.loc[valid, "team"]).sum()
        agg[metrics[col]] = num / den
    return pd.DataFrame(agg)


def team_context(seasons: list[int], weeks: tuple[int, int] = (1, 22),
                 season_type: str | None = "REG") -> pd.DataFrame:
    """Attempt-weighted team NGS profile over the selected window. Empty frame if unavailable."""
    frames = []
    for stat, metrics, weight in (("passing", PASS_METRICS, "attempts"),
                                  ("rushing", RUSH_METRICS, "rush_attempts")):
        raw = data.load_ngs(stat)
        if raw is None:
            continue
        df = raw[raw["season"].isin(seasons) & raw["week"].between(max(weeks[0], 1), weeks[1])]
        if season_type:
            df = df[df["season_type"] == season_type]
        df = df.assign(team=df["team_abbr"].replace(TEAM_FIX))
        frames.append(_weighted(df, metrics, weight))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()
