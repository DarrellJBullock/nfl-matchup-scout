"""Central configuration: data sources, situational dimensions, split definitions, defaults."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "cache"

# nflverse-data release assets (built from nflverse-pbp / ngs-data pipelines).
RELEASE_BASE = "https://github.com/nflverse/nflverse-data/releases/download"
PBP_URL = RELEASE_BASE + "/pbp/play_by_play_{season}.parquet"
PARTICIPATION_URL = RELEASE_BASE + "/pbp_participation/pbp_participation_{season}.parquet"
FTN_URL = RELEASE_BASE + "/ftn_charting/ftn_charting_{season}.parquet"
NGS_URL = RELEASE_BASE + "/nextgen_stats/ngs_{stat}.parquet"  # stat in passing|rushing|receiving

# Current-season files change weekly; refresh them if older than this many hours.
STALE_HOURS = 24

PBP_COLUMNS = [
    "game_id", "play_id", "season", "week", "season_type", "posteam", "defteam",
    "down", "ydstogo", "yardline_100", "play_type", "qb_dropback", "qb_scramble",
    "shotgun", "run_location", "run_gap", "epa", "success", "wp",
    "half_seconds_remaining", "two_point_attempt",
]
PARTICIPATION_COLUMNS = [
    "nflverse_game_id", "play_id", "offense_formation", "offense_personnel",
    "defense_personnel", "defenders_in_box", "number_of_pass_rushers",
    "was_pressure", "defense_man_zone_type",
]
FTN_COLUMNS = ["nflverse_game_id", "nflverse_play_id", "is_play_action", "is_screen_pass", "is_rpo"]


# --- Situational dimensions -------------------------------------------------------------
# Each dimension is a column produced by features.add_dimensions(). Values are human-readable
# labels so a situation key reads naturally in a scouting report. A value of NA means the
# dimension does not apply to that play (e.g. run direction on a pass) or data is missing.
DIMENSIONS: dict[str, str] = {
    "play_family": "Play type",
    "down_dist": "Down & distance",
    "personnel": "Offensive personnel",
    "formation": "Formation",
    "run_dir": "Run direction",
    "def_personnel": "Defensive personnel",
    "box": "Box count",
    "pressure": "Pressure",
    "rushers": "Pass rush",
    "coverage": "Coverage shell",
    "play_action": "Play action",
    "field_zone": "Field zone",
}


@dataclass(frozen=True)
class Split:
    """A family of situations: every observed combination of values across `dims`."""

    key: str
    dims: tuple[str, ...]
    group: str  # for UI grouping

    @property
    def label(self) -> str:
        return " × ".join(DIMENSIONS[d] for d in self.dims)


SPLITS: tuple[Split, ...] = (
    # Down / distance
    Split("dd", ("down_dist",), "Down & distance"),
    Split("dd_pf", ("down_dist", "play_family"), "Down & distance"),
    # Personnel
    Split("pers", ("personnel",), "Personnel"),
    Split("pers_pf", ("personnel", "play_family"), "Personnel"),
    Split("pers_def", ("personnel", "def_personnel", "play_family"), "Personnel"),
    Split("defpers_pf", ("def_personnel", "play_family"), "Personnel"),
    # Formation
    Split("form_pf", ("formation", "play_family"), "Formation"),
    Split("form_run", ("formation", "run_dir"), "Formation"),
    # Run game
    Split("rundir", ("run_dir",), "Run game"),
    Split("pers_rundir", ("personnel", "run_dir"), "Run game"),
    Split("rundir_def", ("run_dir", "def_personnel"), "Run game"),
    Split("pers_rundir_def", ("personnel", "run_dir", "def_personnel"), "Run game"),
    Split("rundir_box", ("run_dir", "box"), "Run game"),
    # Pass game / pressure
    Split("pressure", ("pressure",), "Pressure"),
    Split("pressure_dd", ("pressure", "down_dist"), "Pressure"),
    Split("rushers", ("rushers",), "Pressure"),
    Split("coverage", ("coverage",), "Pass game"),
    Split("pa", ("play_action",), "Pass game"),
    Split("pers_cov", ("personnel", "coverage"), "Pass game"),
    # Red zone
    Split("rz_pf", ("field_zone", "play_family"), "Red zone"),
    Split("rz_pers", ("field_zone", "personnel", "play_family"), "Red zone"),
)
SPLITS_BY_KEY = {s.key: s for s in SPLITS}


@dataclass(frozen=True)
class ProfileSettings:
    min_plays: int = 15          # team needs this many plays in a situation to be flagged
    z_threshold: float = 1.0     # |z| at or above this is a strength / weakness
    epa_weight: float = 0.65     # composite z = w * z_epa + (1 - w) * z_success
    stabilization_plays: float = 20.0  # k league-average plays blended into each team rate (0 = raw)
    wp_min: float = 0.05         # drop garbage-time plays outside this win-prob window
    wp_max: float = 0.95


DEFAULT_SETTINGS = ProfileSettings()
