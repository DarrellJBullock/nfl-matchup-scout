"""Download + cache nflverse release files and assemble a play-level frame."""
from __future__ import annotations

import datetime as dt
import logging
import time
from pathlib import Path

import pandas as pd
import requests

from . import config

log = logging.getLogger(__name__)


def current_season(today: dt.date | None = None) -> int:
    """NFL season year: games in Jan/Feb belong to the previous season."""
    today = today or dt.date.today()
    return today.year if today.month >= 3 else today.year - 1


def _is_stale(path: Path, season: int | None) -> bool:
    if not path.exists():
        return True
    # Completed seasons never change; only the in-progress season needs refreshing.
    if season is not None and season < current_season():
        return False
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    return age_hours > config.STALE_HOURS


def fetch(url: str, season: int | None = None, cache_dir: Path = config.CACHE_DIR) -> Path | None:
    """Download `url` into the cache (if missing/stale). Returns None if the asset doesn't exist."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / url.rsplit("/", 1)[-1]
    if not _is_stale(path, season):
        return path
    try:
        resp = requests.get(url, timeout=120)
    except requests.RequestException as exc:
        log.warning("download failed for %s: %s", url, exc)
        return path if path.exists() else None
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    tmp = path.with_suffix(".part")
    tmp.write_bytes(resp.content)
    tmp.replace(path)
    return path


def _read(url: str, season: int | None, columns: list[str]) -> pd.DataFrame | None:
    path = fetch(url, season)
    if path is None:
        return None
    df = pd.read_parquet(path)
    return df[[c for c in columns if c in df.columns]]


def load_season(season: int) -> tuple[pd.DataFrame, dict[str, bool]]:
    """Play-by-play for one season joined with participation + FTN charting when available."""
    pbp = _read(config.PBP_URL.format(season=season), season, config.PBP_COLUMNS)
    if pbp is None:
        raise FileNotFoundError(f"No nflverse play-by-play published for {season}")

    coverage = {"participation": False, "ftn": False}

    part = _read(config.PARTICIPATION_URL.format(season=season), season, config.PARTICIPATION_COLUMNS)
    if part is not None and len(part):
        part = part.rename(columns={"nflverse_game_id": "game_id"})
        pbp = pbp.merge(part, on=["game_id", "play_id"], how="left")
        coverage["participation"] = True

    ftn = _read(config.FTN_URL.format(season=season), season, config.FTN_COLUMNS)
    if ftn is not None and len(ftn):
        ftn = ftn.rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
        ftn = ftn.drop_duplicates(["game_id", "play_id"])
        pbp = pbp.merge(ftn, on=["game_id", "play_id"], how="left")
        coverage["ftn"] = True

    return pbp, coverage


def load_plays(seasons: list[int]) -> tuple[pd.DataFrame, dict[int, dict[str, bool]]]:
    frames, coverage = [], {}
    for season in seasons:
        df, cov = load_season(season)
        frames.append(df)
        coverage[season] = cov
    return pd.concat(frames, ignore_index=True), coverage


def load_ngs(stat: str) -> pd.DataFrame | None:
    """Next Gen Stats weekly/season aggregates (nflverse ngs-data) — one file for all seasons."""
    path = fetch(config.NGS_URL.format(stat=stat), season=current_season())
    return pd.read_parquet(path) if path is not None else None
