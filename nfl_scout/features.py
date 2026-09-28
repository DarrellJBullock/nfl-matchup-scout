"""Turn raw play-by-play into scrimmage plays tagged with situational dimensions."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import DEFAULT_SETTINGS, ProfileSettings

_COUNT_RE = re.compile(r"(\d+)\s+([A-Z]+)")

OL_POS = {"C", "G", "T", "OL", "OT", "OG"}
DB_POS = {"CB", "FS", "SS", "S", "DB", "SAF"}


def _counts(personnel: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for n, pos in _COUNT_RE.findall(personnel or ""):
        counts[pos] = counts.get(pos, 0) + int(n)
    return counts


def offense_grouping(personnel: object) -> object:
    """'1 C, 2 G, 1 QB, 1 RB, 2 T, 2 TE, 2 WR' -> '12 personnel'. Extra OL is called out."""
    if not isinstance(personnel, str) or not personnel.strip():
        return pd.NA
    c = _counts(personnel)
    rb = c.get("RB", 0) + c.get("FB", 0)
    te = c.get("TE", 0)
    if rb > 9 or te > 9:
        return pd.NA
    label = f"{rb}{te} personnel"
    if sum(c.get(p, 0) for p in OL_POS) >= 6:
        label += " (6 OL)"
    return label


def defense_grouping(personnel: object) -> object:
    """Classify by defensive-back count: base / nickel / dime / heavy."""
    if not isinstance(personnel, str) or not personnel.strip():
        return pd.NA
    dbs = sum(_counts(personnel).get(p, 0) for p in DB_POS)
    if dbs <= 3:
        return "Heavy D (≤3 DB)"
    if dbs == 4:
        return "Base D (4 DB)"
    if dbs == 5:
        return "Nickel D (5 DB)"
    return "Dime D (6+ DB)"


def _down_dist(down: pd.Series, togo: pd.Series) -> pd.Series:
    bucket = np.select([togo >= 7, togo >= 3], ["long (7+)", "medium (3-6)"], "short (1-2)")
    out = np.where(
        down == 1,
        np.where(togo >= 10, "1st & 10+", "1st & <10"),
        np.where(down == 2, "2nd & " + pd.Series(bucket, index=down.index),
                 "3rd/4th & " + pd.Series(bucket, index=down.index)),
    )
    return pd.Series(out, index=down.index, dtype="string")


_FORMATIONS = {
    "SHOTGUN": "Shotgun", "EMPTY": "Shotgun", "PISTOL": "Pistol",
    "UNDER CENTER": "Under center", "SINGLEBACK": "Under center",
    "I_FORM": "Under center", "JUMBO": "Under center",
}


def _where(mask: pd.Series, values: pd.Series | np.ndarray) -> pd.Series:
    s = pd.Series(values, index=mask.index, dtype="string")
    return s.where(mask.fillna(False).astype(bool), pd.NA)


def scrimmage_plays(pbp: pd.DataFrame, settings: ProfileSettings = DEFAULT_SETTINGS) -> pd.DataFrame:
    """Designed runs + dropbacks with valid EPA, excluding 2-pt tries and garbage time."""
    df = pbp[
        pbp["play_type"].isin(["pass", "run"])
        & pbp["epa"].notna()
        & pbp["down"].notna()
        & pbp["posteam"].notna()
        & (pbp["two_point_attempt"].fillna(0) == 0)
    ]
    if settings.wp_min > 0 or settings.wp_max < 1:
        df = df[df["wp"].between(settings.wp_min, settings.wp_max)]
    return df.copy()


def add_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    """Attach every situational dimension from config.DIMENSIONS as a labelled string column."""
    out = df.copy()
    is_pass = out["qb_dropback"].fillna(0).astype(int) == 1
    is_run = ~is_pass
    out["play_family"] = pd.Series(np.where(is_pass, "Pass", "Run"), index=out.index, dtype="string")
    out["down_dist"] = _down_dist(out["down"].astype(int), out["ydstogo"].astype(int))
    out["field_zone"] = _where(
        out["yardline_100"] <= 20,
        np.where(out["yardline_100"] <= 5, "Goal line (inside 5)", "Red zone (6-20)"),
    )

    # Run direction (designed runs only). run_gap: end / tackle / guard; middle has no gap.
    gap = out["run_gap"].astype(object)
    loc = out["run_location"].astype(object)
    run_dir = np.select(
        [(gap == "end").to_numpy(), (gap == "tackle").to_numpy(), ((gap == "guard") | (loc == "middle")).to_numpy()],
        ["Outside run (edge)", "Off-tackle run", "Inside run"],
        default="",
    )
    out["run_dir"] = _where(is_run & (run_dir != ""), run_dir)

    # Formation: charted formation when present, otherwise the pbp shotgun flag.
    charted = out.get("offense_formation", pd.Series(pd.NA, index=out.index)).map(_FORMATIONS)
    fallback = out["shotgun"].map({1: "Shotgun", 0: "Under center"})
    out["formation"] = charted.fillna(fallback).astype("string")

    # Personnel (participation data)
    off_p = out.get("offense_personnel", pd.Series(pd.NA, index=out.index))
    def_p = out.get("defense_personnel", pd.Series(pd.NA, index=out.index))
    out["personnel"] = off_p.map(offense_grouping).astype("string")
    out["def_personnel"] = def_p.map(defense_grouping).astype("string")

    box = pd.to_numeric(out.get("defenders_in_box"), errors="coerce").replace(0, np.nan)
    out["box"] = _where(
        box.notna(),
        np.select([box <= 6, box == 7], ["Light box (≤6)", "7-man box"], "Stacked box (8+)"),
    )

    # Pass-game dimensions (dropbacks only)
    pressure = out.get("was_pressure", pd.Series(pd.NA, index=out.index))
    out["pressure"] = _where(
        is_pass & pressure.notna(),
        np.where(pressure.fillna(False).astype(bool), "Under pressure", "Clean pocket"),
    )
    rushers = pd.to_numeric(out.get("number_of_pass_rushers"), errors="coerce").replace(0, np.nan)
    out["rushers"] = _where(
        is_pass & rushers.notna(),
        np.select([rushers <= 3, rushers == 4], ["3-man rush", "4-man rush"], "Blitz (5+ rushers)"),
    )
    shell = out.get("defense_man_zone_type", pd.Series(pd.NA, index=out.index)).map(
        {"MAN_COVERAGE": "Man coverage", "ZONE_COVERAGE": "Zone coverage"}
    )
    out["coverage"] = _where(is_pass & shell.notna(), shell.fillna(""))
    pa = out.get("is_play_action", pd.Series(pd.NA, index=out.index))
    out["play_action"] = _where(
        is_pass & pa.notna(),
        np.where(pa.fillna(False).astype(bool), "Play-action pass", "Standard dropback"),
    )
    return out


def prepare(pbp: pd.DataFrame, settings: ProfileSettings = DEFAULT_SETTINGS) -> pd.DataFrame:
    return add_dimensions(scrimmage_plays(pbp, settings))
