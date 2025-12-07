import pandas as pd
import pytest

from src.bookmaker_benchmark import compute_team_strength


def test_compute_team_strength_basic():
    """
    compute_team_strength must correctly calculate average win probability
    using pnorm_home_win (home) and pnorm_away_win (away).
    """

    # 1) Fake minimal EPL data
    df = pd.DataFrame({
        "home_team": ["Arsenal", "Chelsea"],
        "away_team": ["Chelsea", "Arsenal"],
        "pnorm_home_win": [0.60, 0.40],   # Arsenal home vs Chelsea home
        "pnorm_away_win": [0.35, 0.20],   # Chelsea away vs Arsenal away
    })

    # 2) Compute strength
    strength = compute_team_strength(df)

    # 3) Must contain exactly 2 teams
    assert set(strength["team"]) == {"Arsenal", "Chelsea"}

    # 4) Check Arsenal values
    arsenal_row = strength[strength["team"] == "Arsenal"].iloc[0]
    # Arsenal home win prob = 0.60 (1 match)
    # Arsenal away win prob = 0.20 (1 match)
    expected_arsenal_strength = (0.60 + 0.20) / 2
    assert pytest.approx(arsenal_row["bet365_strength"], rel=1e-5) == expected_arsenal_strength

    # 5) Check Chelsea values
    chelsea_row = strength[strength["team"] == "Chelsea"].iloc[0]
    expected_chelsea_strength = (0.40 + 0.35) / 2
    assert pytest.approx(chelsea_row["bet365_strength"], rel=1e-5) == expected_chelsea_strength
