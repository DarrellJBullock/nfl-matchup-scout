"""One-call pipeline: seasons -> tagged plays -> team situational profiles."""
from __future__ import annotations

import pandas as pd

from . import data, features, profiles
from .config import DEFAULT_SETTINGS, ProfileSettings


def load_tagged_plays(seasons: list[int], season_types: tuple[str, ...] = ("REG",),
                      weeks: tuple[int, int] = (1, 22),
                      settings: ProfileSettings = DEFAULT_SETTINGS) -> tuple[pd.DataFrame, dict]:
    pbp, coverage = data.load_plays(seasons)
    pbp = pbp[pbp["season_type"].isin(season_types) & pbp["week"].between(*weeks)]
    return features.prepare(pbp, settings), coverage


def run(seasons: list[int], season_types: tuple[str, ...] = ("REG",), weeks: tuple[int, int] = (1, 22),
        settings: ProfileSettings = DEFAULT_SETTINGS) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    plays, coverage = load_tagged_plays(seasons, season_types, weeks, settings)
    return plays, profiles.build_profiles(plays, settings), coverage
