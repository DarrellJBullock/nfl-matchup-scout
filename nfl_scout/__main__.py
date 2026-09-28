"""CLI: python -m nfl_scout KC LV --seasons 2024 2025 > report.md"""
from __future__ import annotations

import argparse

from . import data, matchup, pipeline
from .config import ProfileSettings


def main() -> None:
    ap = argparse.ArgumentParser(description="Print a Markdown matchup scouting report.")
    ap.add_argument("team_a")
    ap.add_argument("team_b")
    ap.add_argument("--seasons", type=int, nargs="+", default=[data.current_season() - 1])
    ap.add_argument("--z", type=float, default=1.0, help="strength/weakness |z| threshold")
    ap.add_argument("--min-plays", type=int, default=15)
    ap.add_argument("--top", type=int, default=6)
    args = ap.parse_args()

    settings = ProfileSettings(z_threshold=args.z, min_plays=args.min_plays)
    _, prof, _ = pipeline.run(args.seasons, settings=settings)
    print(matchup.scouting_report(prof, args.team_a.upper(), args.team_b.upper(), settings, top=args.top))


if __name__ == "__main__":
    main()
