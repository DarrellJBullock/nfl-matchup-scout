"""Cross-reference two teams' profiles to surface matchup exploits.

An *attack* finding is a situation where the offense is flagged as a strength AND the opposing
defense is flagged as a weakness in that same situation. An *avoid* finding is the mirror:
offensive weakness meeting a defensive strength.

Ranking:
  magnitude   = |z_off| + |z_def|
  projected   = off_per_game * def_per_game / league_per_game   (log5-style frequency blend)
  leverage    = magnitude * sqrt(min(projected, frequency_cap))
                The cap stops catch-all situations ("all dropbacks", 35/game) from outranking true
                situational tendencies purely on volume — past ~10 snaps it is a base identity.
  epa_swing   = (off edge + def edge, EPA/play vs league, shrunk) * projected   (EPA per game)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DEFAULT_SETTINGS, ProfileSettings

TIERS = ((9.0, "HIGH"), (6.0, "MEDIUM"), (0.0, "LOW"))


@dataclass(frozen=True)
class MatchupSettings:
    partner_threshold: float | None = None  # z needed on the other side; defaults to z_threshold
    min_projected: float = 1.0              # ignore situations expected < this many snaps/game
    collapse_nested: bool = True            # fold more-specific/general variants into one finding
    frequency_cap: float = 10.0             # snaps/game beyond which frequency adds no leverage


def _tier(score: float) -> str:
    return next(label for cut, label in TIERS if score >= cut)


def cross_reference(profiles: pd.DataFrame, off_team: str, def_team: str, mode: str = "attack",
                    settings: ProfileSettings = DEFAULT_SETTINGS,
                    msettings: MatchupSettings = MatchupSettings()) -> pd.DataFrame:
    """Rank situations where `off_team`'s offense meets `def_team`'s defense."""
    thr = settings.z_threshold
    partner = msettings.partner_threshold if msettings.partner_threshold is not None else thr

    off = profiles[(profiles["side"] == "offense") & (profiles["team"] == off_team)]
    dfn = profiles[(profiles["side"] == "defense") & (profiles["team"] == def_team)]
    m = off.merge(dfn, on=["split", "key"], suffixes=("_off", "_def"))
    m = m[(m["n_off"] >= settings.min_plays) & (m["n_def"] >= settings.min_plays)]

    if mode == "attack":   # offense strong, defense weak (defense z is negative = bad for them)
        m = m[(m["z_off"] >= thr) & (m["z_def"] <= -partner)]
    elif mode == "avoid":  # offense weak, defense strong
        m = m[(m["z_off"] <= -thr) & (m["z_def"] >= partner)]
    else:
        raise ValueError(f"unknown mode {mode!r}")
    if m.empty:
        return pd.DataFrame()

    m = m.assign(
        magnitude=m["z_off"].abs() + m["z_def"].abs(),
        hist_per_game=m["per_game_off"],
        projected=(m["per_game_off"] * m["per_game_def"] / m["lg_per_game_off"]).clip(upper=m[["per_game_off", "per_game_def"]].max(axis=1) * 1.5),
    )
    m = m[m["projected"] >= msettings.min_projected]
    # EPA/play edge for the offense: its own surplus + what the defense concedes above average.
    edge = (m["epa_adj_off"] - m["lg_epa_off"]) + (m["epa_adj_def"] - m["lg_epa_def"])
    m = m.assign(
        leverage=m["magnitude"] * np.sqrt(m["projected"].clip(upper=msettings.frequency_cap)),
        epa_edge_per_play=edge,
        epa_swing_per_game=edge * m["projected"],
    ).sort_values("leverage", ascending=False)

    out = pd.DataFrame({
        "situation": m["situation_off"],
        "split": m["split"],
        "split_label": m["split_label_off"],
        "group": m["group_off"],
        "key": m["key"],
        "n_dims": m["n_dims_off"],
        "off_team": off_team, "def_team": def_team, "mode": mode,
        "off_n": m["n_off"], "off_epa": m["epa_off"], "off_sr": m["sr_off"], "off_z": m["z_off"],
        "def_n": m["n_def"], "def_epa": m["epa_def"], "def_sr": m["sr_def"], "def_z": m["z_def"],
        "lg_epa": m["lg_epa_off"], "lg_sr": m["lg_sr_off"],
        "hist_per_game": m["hist_per_game"], "def_per_game": m["per_game_def"],
        "projected": m["projected"], "magnitude": m["magnitude"], "leverage": m["leverage"],
        "epa_edge_per_play": m["epa_edge_per_play"], "epa_swing_per_game": m["epa_swing_per_game"],
    }).reset_index(drop=True)
    out["tier"] = out["leverage"].map(_tier)
    out["related"] = [[] for _ in range(len(out))]
    if msettings.collapse_nested:
        out = collapse_nested(out)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def collapse_nested(findings: pd.DataFrame) -> pd.DataFrame:
    """Fold findings whose situation is a strict superset/subset of a higher-ranked one.

    e.g. "Outside run (edge)" and "12 personnel · Outside run (edge) · Base D (4 DB)" describe the
    same exploit; the lower-leverage one is listed under `related` of the stronger one.
    """
    keep: list[int] = []
    pairs = [frozenset(k.split("|")) for k in findings["key"]]
    related: dict[int, list[str]] = {}
    for i in range(len(findings)):  # already sorted by leverage desc
        parent = next((j for j in keep if pairs[i] < pairs[j] or pairs[j] < pairs[i]), None)
        if parent is None:
            keep.append(i)
            related[i] = []
        else:
            related[parent].append(findings.at[i, "situation"])
    out = findings.loc[keep].copy()
    out["related"] = [related[i] for i in keep]
    return out.reset_index(drop=True)


def _fmt_epa(x: float) -> str:
    return f"{x:+.2f}"


def describe(f: pd.Series) -> str:
    """One scouting-report paragraph for a finding."""
    off, dfn = f["off_team"], f["def_team"]
    if f["mode"] == "attack":
        headline = f"{f['tier']}-LEVERAGE EXPLOIT — {off} offense: {f['situation']} vs {dfn} defense"
        body = (
            f"{off} averages {_fmt_epa(f['off_epa'])} EPA/play with a {f['off_sr']:.0%} success rate here "
            f"(z {f['off_z']:+.1f}, n={int(f['off_n'])}), while {dfn} allows {_fmt_epa(f['def_epa'])} EPA/play "
            f"and {f['def_sr']:.0%} success (z {f['def_z']:+.1f}, n={int(f['def_n'])}). "
            f"League baseline: {_fmt_epa(f['lg_epa'])} EPA/play, {f['lg_sr']:.0%} success."
        )
    else:
        headline = f"{f['tier']}-LEVERAGE CAUTION — {off} offense: {f['situation']} vs {dfn} defense"
        body = (
            f"{off} struggles here at {_fmt_epa(f['off_epa'])} EPA/play, {f['off_sr']:.0%} success "
            f"(z {f['off_z']:+.1f}, n={int(f['off_n'])}), and {dfn} holds opponents to {_fmt_epa(f['def_epa'])} "
            f"EPA/play, {f['def_sr']:.0%} success (z {f['def_z']:+.1f}, n={int(f['def_n'])}). "
            f"League baseline: {_fmt_epa(f['lg_epa'])} EPA/play."
        )
    freq = (
        f" Occurs about {f['hist_per_game']:.1f} times per game for {off} historically "
        f"(~{f['projected']:.1f} projected vs {dfn}); regressed edge worth ≈{f['epa_swing_per_game']:+.1f} EPA/game."
    )
    related = f" Related splits: {'; '.join(f['related'][:4])}." if f["related"] else ""
    return f"**{headline}.** {body}{freq}{related}"


def scouting_report(profiles: pd.DataFrame, team_a: str, team_b: str,
                    settings: ProfileSettings = DEFAULT_SETTINGS,
                    msettings: MatchupSettings = MatchupSettings(), top: int = 6) -> str:
    """Markdown scouting report covering both sides of the ball for both teams."""
    lines = [f"# Matchup scouting report: {team_a} vs {team_b}", ""]
    for off, dfn in ((team_a, team_b), (team_b, team_a)):
        for mode, title in (("attack", f"{off} offense — attack points vs {dfn}"),
                            ("avoid", f"{off} offense — tendencies to avoid vs {dfn}")):
            lines.append(f"## {title}")
            found = cross_reference(profiles, off, dfn, mode, settings, msettings).head(top)
            if found.empty:
                lines.append("_No overlapping flags at the current thresholds._")
            for _, f in found.iterrows():
                lines.append(f"{int(f['rank'])}. {describe(f)}")
            lines.append("")
    return "\n".join(lines)
