"""Synthetic-data tests for tagging, profiling and exploit cross-referencing (no network)."""
import numpy as np
import pandas as pd
import pytest

from nfl_scout import features, matchup, profiles
from nfl_scout.config import ProfileSettings, Split

TEAMS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]


def test_offense_grouping():
    assert features.offense_grouping("1 C, 2 G, 1 QB, 1 RB, 2 T, 2 TE, 2 WR") == "12 personnel"
    assert features.offense_grouping("1 RB, 1 TE, 3 WR") == "11 personnel"
    assert features.offense_grouping("1 C, 2 G, 1 QB, 1 RB, 3 T, 1 TE, 2 WR") == "11 personnel (6 OL)"
    assert features.offense_grouping("1 QB, 1 FB, 1 RB, 1 TE, 2 WR, 5 OL") == "21 personnel"
    assert features.offense_grouping(None) is pd.NA


def test_defense_grouping():
    assert features.defense_grouping("2 CB, 2 DE, 2 DT, 1 FS, 2 ILB, 1 SS") == "Base D (4 DB)"
    assert features.defense_grouping("3 CB, 2 DE, 2 DT, 1 FS, 2 ILB, 1 SS") == "Nickel D (5 DB)"
    assert features.defense_grouping("4 CB, 2 DE, 1 DT, 1 FS, 1 ILB, 2 SS") == "Dime D (6+ DB)"


def _raw_plays(seed=0, n_games=12):
    """Round-robin-ish schedule; AAA's 12-personnel edge runs are elite and BBB defends them badly."""
    rng = np.random.default_rng(seed)
    rows, pid = [], 0
    for g in range(n_games):
        for i, off in enumerate(TEAMS):
            dfn = TEAMS[(i + 1 + g) % len(TEAMS)]
            if dfn == off:
                dfn = TEAMS[(i + 2 + g) % len(TEAMS)]
            game_id = f"G{g}_{min(off, dfn)}_{max(off, dfn)}"
            for _ in range(40):
                pid += 1
                run = rng.random() < 0.45
                pers = "1 RB, 2 TE, 2 WR" if rng.random() < 0.4 else "1 RB, 1 TE, 3 WR"
                gap = rng.choice(["end", "tackle", "guard"]) if run else None
                epa = rng.normal(0.0, 1.0)
                if run and gap == "end" and pers.startswith("1 RB, 2 TE"):
                    epa += 0.9 if off == "AAA" else 0.0
                    epa += 0.9 if dfn == "BBB" else 0.0
                rows.append(dict(
                    game_id=game_id, play_id=pid, season=2025, week=g + 1, season_type="REG",
                    posteam=off, defteam=dfn, down=int(rng.integers(1, 4)), ydstogo=int(rng.integers(1, 15)),
                    yardline_100=int(rng.integers(1, 99)), play_type="run" if run else "pass",
                    qb_dropback=0 if run else 1, qb_scramble=0, shotgun=int(not run),
                    run_location=("left" if gap else None), run_gap=gap, epa=epa, success=float(epa > 0),
                    wp=0.5, half_seconds_remaining=900, two_point_attempt=0,
                    offense_personnel=pers, defense_personnel="2 CB, 2 DE, 2 DT, 1 FS, 2 ILB, 1 SS",
                    offense_formation="UNDER CENTER" if run else "SHOTGUN", defenders_in_box=7,
                    number_of_pass_rushers=0 if run else 4, was_pressure=False,
                    defense_man_zone_type="" if run else "ZONE_COVERAGE", is_play_action=False,
                ))
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def plays():
    return features.prepare(_raw_plays())


SPLIT = Split("pers_rundir", ("personnel", "run_dir"), "Run game")
SITUATION = "12 personnel · Outside run (edge)"


def test_dimensions_tagging(plays):
    runs = plays[plays["play_family"] == "Run"]
    passes = plays[plays["play_family"] == "Pass"]
    assert runs["run_dir"].notna().all() and passes["run_dir"].isna().all()
    assert passes["coverage"].eq("Zone coverage").all() and runs["coverage"].isna().all()
    assert set(plays["def_personnel"].dropna()) == {"Base D (4 DB)"}


def test_profiles_flag_planted_strength_and_weakness(plays):
    settings = ProfileSettings(min_plays=10, z_threshold=1.0)
    prof = profiles.build_profiles(plays, settings, splits=(SPLIT,))
    row = lambda side, team: prof[(prof.side == side) & (prof.team == team) & (prof.situation == SITUATION)].iloc[0]
    off, dfn = row("offense", "AAA"), row("defense", "BBB")
    assert off["z"] > 1.5 and off["flag"] == "Strength"
    assert dfn["z"] < -1.5 and dfn["flag"] == "Weakness"      # defense z oriented: negative = bad
    # z-scores are centred on the league in each situation
    sit = prof[(prof.situation == SITUATION) & (prof.side == "offense")]
    assert abs(sit["z"].mean()) < 0.6


def test_stabilization_pulls_small_samples_toward_league(plays):
    raw = profiles.build_profiles(plays, ProfileSettings(stabilization_plays=0), splits=(SPLIT,))
    reg = profiles.build_profiles(plays, ProfileSettings(stabilization_plays=50), splits=(SPLIT,))
    dev_raw = (raw["epa_adj"] - raw["lg_epa"]).abs().mean()
    dev_reg = (reg["epa_adj"] - reg["lg_epa"]).abs().mean()
    assert dev_reg < dev_raw


def test_cross_reference_finds_planted_exploit(plays):
    settings = ProfileSettings(min_plays=10, z_threshold=1.0)
    prof = profiles.build_profiles(plays, settings)
    found = matchup.cross_reference(prof, "AAA", "BBB", "attack", settings)
    assert not found.empty
    top = found.iloc[0]
    assert "Outside run (edge)" in top["situation"]
    assert top["off_z"] >= 1.0 and top["def_z"] <= -1.0
    assert top["projected"] > 0 and top["leverage"] == pytest.approx(
        top["magnitude"] * np.sqrt(min(top["projected"], 10.0)))
    text = matchup.describe(top)
    assert "AAA" in text and "BBB" in text and "times per game" in text


def test_collapse_nested_folds_variants():
    df = pd.DataFrame({
        "key": ["run_dir=Edge", "personnel=12|run_dir=Edge", "coverage=Man"],
        "situation": ["Edge", "12 · Edge", "Man"], "leverage": [9.0, 8.0, 7.0],
    })
    out = matchup.collapse_nested(df)
    assert list(out["situation"]) == ["Edge", "Man"]
    assert out.loc[0, "related"] == ["12 · Edge"]


def test_avoid_mode_and_bad_mode(plays):
    settings = ProfileSettings(min_plays=10)
    prof = profiles.build_profiles(plays, settings, splits=(SPLIT,))
    assert isinstance(matchup.cross_reference(prof, "AAA", "BBB", "avoid", settings), pd.DataFrame)
    with pytest.raises(ValueError):
        matchup.cross_reference(prof, "AAA", "BBB", "nope", settings)
