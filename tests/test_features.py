import pandas as pd
from laliga_pipeline.features import to_long, rolling_form

def test_long_and_rolling():
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2024-08-15"]),
        "Season": ["2024-25"],
        "HomeTeam":["barcelona"],"AwayTeam":["real madrid"],
        "FTHG":[13],"FTAG":[14],
        "HS":[15],"HST":[16],"HF":[17],"HC":[16],"HY":[13],"HR":[0],
        "AS":[18],"AST":[19],"AF":[15],"AC":[20],"AY":[20],"AR":[0],
        "FTR":["H"]
    })
    long = to_long(df)
    feat = rolling_form(long, windows=[14])
    assert len(long)==2 and not feat.empty
