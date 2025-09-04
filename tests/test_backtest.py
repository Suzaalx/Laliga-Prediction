import pandas as pd
from laliga_pipeline.backtest import rolling_origin

def test_rolling_origin_metrics():
    d = pd.DataFrame({
        "Date": pd.to_datetime(["2024-08-01","2024-08-08","2024-08-15","2024-08-22","2024-08-29"]),
        "HomeTeam": ["a","b","a","b","a"],
        "AwayTeam": ["b","a","b","a","b"],
        "FTHG":[1,2,0,1,2],
        "FTAG":[0,1,1,1,1],
        "FTR":["H","H","A","D","H"]
    })
    m = rolling_origin(d, min_train=3, xi=0.001, max_goals=6)
    assert m["n"] == 2 and m["log_loss"] > 0
