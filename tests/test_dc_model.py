import pandas as pd
from laliga_pipeline.dc_model import fit_dc

def test_fit_dc_small():
    d = pd.DataFrame({
        "Date": pd.to_datetime(["2024-08-15","2024-08-22","2024-08-29","2024-09-05"]),
        "HomeTeam": ["a","b","a","b"],
        "AwayTeam": ["b","a","b","a"],
        "FTHG":[1,2,0,1],
        "FTAG":[0,1,1,1],
        "FTR": ["H","H","A","D"]
    })
    params = fit_dc(d.iloc[:3], xi=0.001)
    assert set(["home_adv","rho"]).issubset(params.keys())
