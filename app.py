"""Streamlit front end: team situational profiles + matchup exploit finder.

Run:  streamlit run app.py
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from nfl_scout import data, matchup, ngs, pipeline
from nfl_scout.config import DIMENSIONS, SPLITS, SPLITS_BY_KEY, ProfileSettings
from nfl_scout.matchup import MatchupSettings
from nfl_scout.profiles import build_profiles, team_report

st.set_page_config(page_title="NFL Matchup Scout", page_icon="🏈", layout="wide")

FLAG_ICON = {"Strength": "🟢 Strength", "Weakness": "🔴 Weakness", "Neutral": "⚪ Neutral", "Low sample": "· Low sample"}
TIER_ICON = {"HIGH": "🔥 HIGH", "MEDIUM": "⚠️ MEDIUM", "LOW": "· LOW"}


# ---------------------------------------------------------------- cached computation
@st.cache_data(show_spinner="Downloading & tagging nflverse play-by-play…")
def get_plays(seasons: tuple[int, ...], season_types: tuple[str, ...], weeks: tuple[int, int],
              wp_min: float, wp_max: float):
    settings = ProfileSettings(wp_min=wp_min, wp_max=wp_max)
    return pipeline.load_tagged_plays(list(seasons), season_types, weeks, settings)


@st.cache_data(show_spinner="Profiling every team across situational splits…")
def get_profiles(seasons, season_types, weeks, wp_min, wp_max, min_plays, z_threshold, epa_weight, k):
    plays, coverage = get_plays(seasons, season_types, weeks, wp_min, wp_max)
    settings = ProfileSettings(min_plays=min_plays, z_threshold=z_threshold, epa_weight=epa_weight,
                               stabilization_plays=k, wp_min=wp_min, wp_max=wp_max)
    return build_profiles(plays, settings), coverage, len(plays)


@st.cache_data(show_spinner="Loading Next Gen Stats…")
def get_ngs(seasons: tuple[int, ...], weeks: tuple[int, int], season_type: str | None):
    try:
        return ngs.team_context(list(seasons), weeks, season_type)
    except Exception:  # NGS is optional context; never block the app on it
        return pd.DataFrame()


# ---------------------------------------------------------------- sidebar
cur = data.current_season()
with st.sidebar:
    st.header("Data window")
    seasons = st.multiselect("Seasons", list(range(cur, 2015, -1)), default=[cur - 1],
                             help="Personnel, formation, pressure, box & coverage come from nflverse "
                                  "participation data, which lags play-by-play. Splits that need it "
                                  "drop out for seasons where it isn't published yet.")
    include_post = st.checkbox("Include playoffs", value=False)
    weeks = st.slider("Weeks", 1, 22, (1, 22))
    garbage = st.checkbox("Exclude garbage time (win prob outside 5–95%)", value=True)

    st.header("Flagging")
    z_threshold = st.slider("Strength / weakness |z| threshold", 0.5, 2.5, 1.0, 0.1)
    min_plays = st.slider("Min plays per team in a situation", 5, 60, 15, 5)
    k = st.slider("Stabilization (league-average plays blended in)", 0, 100, 20, 5,
                  help="Regresses small samples toward league average before z-scoring. 0 = raw rates.")
    epa_weight = st.slider("Composite weight on EPA (vs success rate)", 0.0, 1.0, 0.65, 0.05)

    st.header("Matchup ranking")
    partner = st.slider("Opponent-side |z| required", 0.0, 2.5, z_threshold, 0.1,
                        help="Lower this to include near-miss overlaps (e.g. strength vs below-average).")
    freq_cap = st.slider("Frequency cap (snaps/game)", 3.0, 30.0, 10.0, 1.0)
    min_proj = st.slider("Min projected snaps/game", 0.0, 5.0, 1.0, 0.5)
    collapse = st.checkbox("Fold nested variants of the same finding", value=True)

if not seasons:
    st.info("Pick at least one season.")
    st.stop()

season_types = ("REG", "POST") if include_post else ("REG",)
wp = (0.05, 0.95) if garbage else (0.0, 1.0)
key = (tuple(sorted(seasons)), season_types, tuple(weeks), *wp)
try:
    prof, coverage, n_plays = get_profiles(*key, min_plays, z_threshold, epa_weight, float(k))
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

settings = ProfileSettings(min_plays=min_plays, z_threshold=z_threshold, epa_weight=epa_weight,
                           stabilization_plays=float(k))
msettings = MatchupSettings(partner_threshold=partner, min_projected=min_proj,
                            collapse_nested=collapse, frequency_cap=freq_cap)
teams = sorted(prof["team"].unique())
ngs_ctx = get_ngs(tuple(sorted(seasons)), tuple(weeks), None if include_post else "REG")

st.title("🏈 NFL Matchup Scout")
missing = [s for s, c in coverage.items() if not c["participation"]]
st.caption(
    f"{n_plays:,} scrimmage plays · seasons {', '.join(map(str, sorted(seasons)))} · "
    f"{len(teams)} teams · {prof['split'].nunique()} split families · "
    f"{prof['situation'].nunique():,} distinct situations"
    + (f" · ⚠️ no participation data for {', '.join(map(str, missing))} (personnel/pressure/coverage splits limited)"
       if missing else "")
)

tab_match, tab_team, tab_league, tab_method = st.tabs(
    ["🎯 Matchup exploits", "📋 Team profile", "🗺️ League heatmap", "📖 Methodology"])


def fmt_findings(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "#": df["rank"],
        "Tier": df["tier"].map(TIER_ICON),
        "Situation": df["situation"],
        "Off EPA/play": df["off_epa"], "Off SR": df["off_sr"], "Off z": df["off_z"],
        "Def EPA allowed": df["def_epa"], "Def SR allowed": df["def_sr"], "Def z": df["def_z"],
        "Lg EPA": df["lg_epa"],
        "Hist /gm": df["hist_per_game"], "Proj /gm": df["projected"],
        "Leverage": df["leverage"], "EPA swing /gm": df["epa_swing_per_game"],
        "Split": df["split_label"],
    })


FINDING_COLS = {
    "Off EPA/play": st.column_config.NumberColumn(format="%+.2f"),
    "Def EPA allowed": st.column_config.NumberColumn(format="%+.2f"),
    "Lg EPA": st.column_config.NumberColumn(format="%+.2f"),
    "Off SR": st.column_config.NumberColumn(format="percent"),
    "Def SR allowed": st.column_config.NumberColumn(format="percent"),
    "Off z": st.column_config.NumberColumn(format="%+.2f"),
    "Def z": st.column_config.NumberColumn(format="%+.2f"),
    "Hist /gm": st.column_config.NumberColumn(format="%.1f"),
    "Proj /gm": st.column_config.NumberColumn(format="%.1f"),
    "Leverage": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=16),
    "EPA swing /gm": st.column_config.NumberColumn(format="%+.1f"),
}


# ---------------------------------------------------------------- matchup tab
with tab_match:
    c1, c2 = st.columns(2)
    team_a = c1.selectbox("Team A", teams, index=teams.index("CHI") if "CHI" in teams else 0)
    team_b = c2.selectbox("Team B", teams, index=teams.index("NYG") if "NYG" in teams else 1)
    if team_a == team_b:
        st.warning("Pick two different teams.")
    else:
        for off, dfn in ((team_a, team_b), (team_b, team_a)):
            st.subheader(f"{off} offense vs {dfn} defense")
            attack = matchup.cross_reference(prof, off, dfn, "attack", settings, msettings)
            avoid = matchup.cross_reference(prof, off, dfn, "avoid", settings, msettings)
            if attack.empty:
                st.caption("No offense-strength × defense-weakness overlaps at these thresholds. "
                           "Try lowering the opponent-side |z| in the sidebar.")
            else:
                for _, f in attack.head(5).iterrows():
                    box = st.error if f["tier"] == "HIGH" else st.warning if f["tier"] == "MEDIUM" else st.info
                    box(matchup.describe(f), icon="🎯")
                with st.expander(f"All {len(attack)} attack points (table)"):
                    st.dataframe(fmt_findings(attack), hide_index=True, column_config=FINDING_COLS,
                                 use_container_width=True)
            with st.expander(f"🛑 {len(avoid)} tendencies for {off} to avoid (offense weakness × {dfn} defense strength)"):
                if avoid.empty:
                    st.caption("None at these thresholds.")
                else:
                    for _, f in avoid.head(5).iterrows():
                        st.markdown(matchup.describe(f))
                    st.dataframe(fmt_findings(avoid), hide_index=True, column_config=FINDING_COLS,
                                 use_container_width=True)
            st.divider()

        if not ngs_ctx.empty and {team_a, team_b} <= set(ngs_ctx.index):
            st.subheader("Next Gen Stats context")
            ctx = ngs_ctx.loc[[team_a, team_b]].T
            ranks = ngs_ctx.rank(ascending=False).loc[[team_a, team_b]].T.astype(int)
            ctx_view = ctx.round(2).astype(str) + "  (#" + ranks.astype(str) + ")"
            st.dataframe(ctx_view, use_container_width=True)
            st.caption("Offensive NGS profile over the selected window; rank is league-wide, descending.")

        report = matchup.scouting_report(prof, team_a, team_b, settings, msettings)
        st.download_button("⬇️ Download full scouting report (Markdown)", report,
                           file_name=f"scouting_{team_a}_vs_{team_b}.md", mime="text/markdown")


# ---------------------------------------------------------------- team tab
with tab_team:
    c1, c2, c3 = st.columns([1, 1, 2])
    team = c1.selectbox("Team", teams, key="profile_team")
    side = c2.radio("Unit", ["offense", "defense"], horizontal=True)
    groups = c3.multiselect("Split groups", sorted({s.group for s in SPLITS}), default=[])
    unit = prof[(prof["team"] == team) & (prof["side"] == side)]
    if groups:
        unit = unit[unit["group"].isin(groups)]

    rep = team_report(unit, team, side, top=12)
    col_s, col_w = st.columns(2)

    def unit_table(df: pd.DataFrame) -> pd.DataFrame:
        epa_lbl = "EPA/play" if side == "offense" else "EPA allowed"
        return pd.DataFrame({
            "Situation": df["situation"], "z": df["z"], epa_lbl: df["epa"], "Lg EPA": df["lg_epa"],
            "SR": df["sr"], "Lg SR": df["lg_sr"], "Plays": df["n"], "/game": df["per_game"],
            "Split": df["split_label"],
        })

    table_cfg = {
        "z": st.column_config.NumberColumn(format="%+.2f"),
        "EPA/play": st.column_config.NumberColumn(format="%+.2f"),
        "EPA allowed": st.column_config.NumberColumn(format="%+.2f"),
        "Lg EPA": st.column_config.NumberColumn(format="%+.2f"),
        "SR": st.column_config.NumberColumn(format="percent"),
        "Lg SR": st.column_config.NumberColumn(format="percent"),
        "/game": st.column_config.NumberColumn(format="%.1f"),
    }
    with col_s:
        st.markdown(f"#### 🟢 {team} {side} strengths")
        st.dataframe(unit_table(rep["strengths"]), hide_index=True, column_config=table_cfg,
                     use_container_width=True)
    with col_w:
        st.markdown(f"#### 🔴 {team} {side} weaknesses")
        st.dataframe(unit_table(rep["weaknesses"]), hide_index=True, column_config=table_cfg,
                     use_container_width=True)

    st.markdown("#### Situational profile by split")
    split_key = st.selectbox("Split", [s.key for s in SPLITS], format_func=lambda k: SPLITS_BY_KEY[k].label)
    one = unit[(unit["split"] == split_key)].sort_values("per_game", ascending=False)
    if one.empty:
        st.caption("No data for this split in the selected window.")
    else:
        chart = alt.Chart(one).mark_bar().encode(
            x=alt.X("z:Q", title="Composite z (+ = good for this unit)", scale=alt.Scale(domain=[-3, 3], clamp=True)),
            y=alt.Y("situation:N", sort="-x", title=None),
            color=alt.Color("flag:N", scale=alt.Scale(
                domain=["Strength", "Neutral", "Weakness", "Low sample"],
                range=["#1a9850", "#9e9e9e", "#d73027", "#e0e0e0"])),
            tooltip=["situation", alt.Tooltip("z:Q", format="+.2f"), alt.Tooltip("epa:Q", format="+.3f"),
                     alt.Tooltip("lg_epa:Q", format="+.3f"), alt.Tooltip("sr:Q", format=".1%"),
                     alt.Tooltip("n:Q", title="plays"), alt.Tooltip("per_game:Q", format=".1f")],
        ).properties(height=max(120, 22 * len(one)))
        st.altair_chart(chart, use_container_width=True)

    if side == "offense" and not ngs_ctx.empty and team in ngs_ctx.index:
        st.markdown("#### Next Gen Stats context")
        st.dataframe(ngs_ctx.loc[[team]].round(2), use_container_width=True)


# ---------------------------------------------------------------- league tab
with tab_league:
    c1, c2 = st.columns([2, 1])
    lk = c1.selectbox("Split", [s.key for s in SPLITS], format_func=lambda k: SPLITS_BY_KEY[k].label,
                      index=[s.key for s in SPLITS].index("pers_rundir"), key="league_split")
    lside = c2.radio("Unit", ["offense", "defense"], horizontal=True, key="league_side")
    grid = prof[(prof["split"] == lk) & (prof["side"] == lside)]
    # Keep the situations the league actually runs, most common first
    common = grid.groupby("situation")["lg_n"].first().sort_values(ascending=False)
    keep = common.head(14).index
    grid = grid[grid["situation"].isin(keep)].assign(
        z_shown=lambda d: d["z"].where(d["n"] >= min_plays))
    if grid.empty:
        st.caption("No data for this split in the selected window.")
    else:
        heat = alt.Chart(grid).mark_rect().encode(
            x=alt.X("situation:N", sort=list(keep), title=None, axis=alt.Axis(labelAngle=-35, labelLimit=220)),
            y=alt.Y("team:N", title=None),
            color=alt.Color("z_shown:Q", title="z", scale=alt.Scale(scheme="redblue", domain=[-2.5, 2.5])),
            tooltip=["team", "situation", alt.Tooltip("z:Q", format="+.2f"), alt.Tooltip("epa:Q", format="+.3f"),
                     alt.Tooltip("lg_epa:Q", format="+.3f"), alt.Tooltip("sr:Q", format=".1%"),
                     alt.Tooltip("n:Q", title="plays"), alt.Tooltip("per_game:Q", format=".1f"), "flag"],
        ).properties(height=32 * len(teams))
        st.altair_chart(heat, use_container_width=True)
        st.caption("Blue = strength, red = weakness (already oriented per unit). Blank = below the min-plays bar. "
                   "Columns ordered by league-wide frequency.")


# ---------------------------------------------------------------- methodology tab
with tab_method:
    st.markdown(f"""
**Data.** nflverse play-by-play (`nflverse-pbp`), joined on `game_id`+`play_id` to nflverse participation
(personnel, formation, box count, pass rushers, pressure, man/zone) and FTN charting (play action),
all pulled from `nflverse-data` releases and cached in `data/cache/`. Next Gen Stats (`ngs-data`) supplies
offensive context (time to throw, aggressiveness, RYOE, 8-man-box rate).

**Plays.** Designed runs and dropbacks (scrambles/sacks count as dropbacks) with valid EPA; two-point tries
excluded; garbage time optionally excluded by win probability.

**Dimensions.** {", ".join(DIMENSIONS.values())}. Run direction uses the charted gap: *edge* (end),
*off-tackle*, *inside* (guard/middle). pbp doesn't label scheme, so edge/off-tackle runs are the closest
proxy for outside zone and stretch concepts. Personnel groups count RB+FB and TE (e.g. `12` = 1 RB, 2 TE);
defensive personnel is classified by DB count.

**Profiles.** {len(SPLITS)} split families (single dimensions and scouting-relevant combinations). For every
(team, unit, situation) we compute EPA/play and success rate, regress them toward the league baseline by
adding *k* league-average plays, then z-score against the other qualified teams in that exact situation.
The composite z is `w·z_EPA + (1−w)·z_SR`, sign-flipped for defenses, so **positive always means good
for that unit**. |z| ≥ threshold ⇒ Strength / Weakness.

**Exploits.** For Offense A vs Defense B we join the two profiles on the identical situation key and keep
rows where A is a strength and B is a weakness (*attack*), or the reverse (*avoid*). Then:

- `magnitude = |z_off| + |z_def|`
- `projected snaps/game = off_rate × def_rate ÷ league_rate` (log5-style: both the play-caller's habit and
  what the defense tends to see)
- `leverage = magnitude × √min(projected, cap)`. The cap stops "all dropbacks" from winning on volume alone.
- `EPA swing/game = (off surplus + def deficit, regressed EPA/play) × projected`

Nested variants of one finding (e.g. *Outside run* ⊂ *12 personnel · Outside run · Base D*) are folded into
the stronger one and listed as related splits.

**Caveats.** These are descriptive tendencies. Situations interact with opponent quality and game script,
and personnel labels come from roster positions (a TE lined up as a fullback counts as TE). Treat
low-sample HIGH findings as film-study leads, not conclusions.
""")
