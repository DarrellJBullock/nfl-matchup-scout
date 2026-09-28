"""Team situational profiles: per-situation EPA / success rate, z-scores vs league, flags.

Method, per situation (one combination of dimension values within a split):

1. League baseline = play-weighted EPA/play and success rate over every play in the situation.
2. Each team's raw rate is regressed toward that baseline by adding k league-average plays:
       adj_i = (n_i * raw_i + k * league) / (n_i + k)
   so a 15-play sample can't masquerade as an elite unit (k = settings.stabilization_plays).
3. z_i = (adj_i - league) / sd_adj, where sd_adj is the RMS deviation of the regressed rates
   of qualified teams (n >= min_plays) from the league baseline in that situation.
4. Composite z = w * z_epa + (1 - w) * z_success, sign-flipped for defenses so that
   positive always means "good for this unit". |z| >= threshold -> Strength / Weakness.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEFAULT_SETTINGS, SPLITS, ProfileSettings, Split

SIDES = {"offense": "posteam", "defense": "defteam"}


def _situation_key(dims: tuple[str, ...], values: tuple[str, ...]) -> str:
    return "|".join(sorted(f"{d}={v}" for d, v in zip(dims, values)))


def _regressed_z(team: pd.DataFrame, raw: pd.Series, league: pd.Series, k: float,
                 min_n: int) -> tuple[pd.Series, pd.Series]:
    """Regress each team's rate toward league with k phantom league-average plays, then z-score
    the regressed rates across qualified teams within the same situation."""
    adjusted = (team["n"] * raw + k * league) / (team["n"] + k)
    dev = adjusted - league
    qualified = team["n"] >= min_n
    sd = np.sqrt(dev.where(qualified).pow(2).groupby(team["sit_id"]).transform("mean"))
    z = (dev / sd.replace(0, np.nan)).fillna(0.0)
    return adjusted, z


def _profile_split(plays: pd.DataFrame, split: Split, games: dict[str, pd.Series],
                   settings: ProfileSettings) -> pd.DataFrame:
    dims = list(split.dims)
    sub = plays.dropna(subset=dims)
    if sub.empty:
        return pd.DataFrame()

    league = sub.groupby(dims, observed=True).agg(
        lg_n=("epa", "size"), lg_epa=("epa", "mean"),
        lg_sr=("success", "mean"),
    ).reset_index()
    league["sit_id"] = np.arange(len(league))

    frames = []
    for side, team_col in SIDES.items():
        team = sub.groupby([team_col, *dims], observed=True).agg(
            n=("epa", "size"), epa=("epa", "mean"), sr=("success", "mean"),
        ).reset_index().rename(columns={team_col: "team"})
        team = team.merge(league, on=dims, how="left")
        team["side"] = side
        team["games"] = team["team"].map(games[side]).astype(float)
        frames.append(team)
    out = pd.concat(frames, ignore_index=True)
    # sit_id must separate offense and defense for between-team variance
    out["sit_id"] = out["sit_id"].astype(str) + ":" + out["side"]

    total_team_games = games["offense"].sum()
    out["per_game"] = out["n"] / out["games"]
    out["lg_per_game"] = out["lg_n"] / total_team_games

    k = settings.stabilization_plays
    out["epa_adj"], z_epa = _regressed_z(out, out["epa"], out["lg_epa"], k, settings.min_plays)
    out["sr_adj"], z_sr = _regressed_z(out, out["sr"], out["lg_sr"], k, settings.min_plays)

    # Defense: allowing less EPA / lower success is good -> flip sign.
    sign = np.where(out["side"] == "defense", -1.0, 1.0)
    out["z_epa"] = sign * z_epa
    out["z_sr"] = sign * z_sr
    out["z"] = settings.epa_weight * out["z_epa"] + (1 - settings.epa_weight) * out["z_sr"]

    values = out[dims].astype(str).itertuples(index=False, name=None)
    out["values"] = list(values)
    out["situation"] = [" · ".join(v) for v in out["values"]]
    out["key"] = [_situation_key(split.dims, v) for v in out["values"]]
    out["split"] = split.key
    out["split_label"] = split.label
    out["group"] = split.group
    out["n_dims"] = len(dims)
    return out.drop(columns=dims + ["sit_id"])


def team_games(plays: pd.DataFrame) -> dict[str, pd.Series]:
    return {side: plays.groupby(col)["game_id"].nunique() for side, col in SIDES.items()}


def build_profiles(plays: pd.DataFrame, settings: ProfileSettings = DEFAULT_SETTINGS,
                   splits: tuple[Split, ...] = SPLITS) -> pd.DataFrame:
    """Long table: one row per (side, team, split, situation)."""
    games = team_games(plays)
    frames = [_profile_split(plays, s, games, settings) for s in splits]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    prof = pd.concat(frames, ignore_index=True)

    qualified = prof["n"] >= settings.min_plays
    prof["flag"] = np.select(
        [~qualified, prof["z"] >= settings.z_threshold, prof["z"] <= -settings.z_threshold],
        ["Low sample", "Strength", "Weakness"],
        default="Neutral",
    )
    cols = ["side", "team", "group", "split", "split_label", "situation", "key", "values", "n_dims",
            "n", "games", "per_game", "lg_per_game", "epa", "epa_adj", "lg_epa", "sr", "sr_adj", "lg_sr",
            "z_epa", "z_sr", "z", "flag", "lg_n"]
    return prof[cols]


def team_report(profiles: pd.DataFrame, team: str, side: str, top: int = 15) -> dict[str, pd.DataFrame]:
    """Top strengths and weaknesses for one unit, strongest signal first."""
    unit = profiles[(profiles["team"] == team) & (profiles["side"] == side)]
    strengths = unit[unit["flag"] == "Strength"].sort_values("z", ascending=False).head(top)
    weaknesses = unit[unit["flag"] == "Weakness"].sort_values("z").head(top)
    return {"strengths": strengths, "weaknesses": weaknesses}
