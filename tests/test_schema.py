import pandas as pd
from laliga_pipeline.data_schema import MatchSchema

def test_schema_validates_minimal_row():
    row = {
        "Date":"15/08/24","HomeTeam":"Barcelona","AwayTeam":"Real Madrid",
        "FTHG":2,"FTAG":1,"FTR":"H","HTHG":1,"HTAG":1,"HTR":"D",
        "Referee":"","HS":10,"AS":8,"HST":5,"AST":4,"HF":12,"AF":10,
        "HC":5,"AC":3,"HY":2,"AY":3,"HR":0,"AR":0
    }
    df = pd.DataFrame([row])
    out = MatchSchema.validate(df)
    assert "Date" in out.columns
