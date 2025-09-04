import pandera as pa
from pandera import Column, Check

MatchSchema = pa.DataFrameSchema(
    {
        "Date": Column(pa.String, nullable=False),
        "HomeTeam": Column(pa.String, nullable=False),
        "AwayTeam": Column(pa.String, nullable=False),
        "FTHG": Column(pa.Int, Check.ge(0), nullable=False),
        "FTAG": Column(pa.Int, Check.ge(0), nullable=False),
        "FTR": Column(pa.String, Check.isin(["H","D","A"]), nullable=True),
        "HTHG": Column(pa.Int, Check.ge(0), nullable=True),
        "HTAG": Column(pa.Int, Check.ge(0), nullable=True),
        "HTR": Column(pa.String, nullable=True),
        "Referee": Column(pa.String, nullable=True),
        "HS": Column(pa.Float, Check.ge(0), nullable=True),
        "AS": Column(pa.Float, Check.ge(0), nullable=True),
        "HST": Column(pa.Float, Check.ge(0), nullable=True),
        "AST": Column(pa.Float, Check.ge(0), nullable=True),
        "HF": Column(pa.Float, Check.ge(0), nullable=True),
        "AF": Column(pa.Float, Check.ge(0), nullable=True),
        "HC": Column(pa.Float, Check.ge(0), nullable=True),
        "AC": Column(pa.Float, Check.ge(0), nullable=True),
        "HY": Column(pa.Float, Check.ge(0), nullable=True),
        "AY": Column(pa.Float, Check.ge(0), nullable=True),
        "HR": Column(pa.Float, Check.ge(0), nullable=True),
        "AR": Column(pa.Float, Check.ge(0), nullable=True),
    },
    coerce=True,
    strict=False,
)
UpcomingFixtureSchema = pa.DataFrameSchema(
    {
        "Date": Column(pa.String, nullable=False),
        "HomeTeam": Column(pa.String, nullable=False),
        "AwayTeam": Column(pa.String, nullable=False),
        "FTHG": Column(pa.String, nullable=True),  # Keep as string for upcoming fixtures
        "FTAG": Column(pa.String, nullable=True),  # Keep as string for upcoming fixtures
        "FTR": Column(pa.String, nullable=True),
        "HTHG": Column(pa.String, nullable=True),
        "HTAG": Column(pa.String, nullable=True),
        "HTR": Column(pa.String, nullable=True),
        "Referee": Column(pa.String, nullable=True),
        "HS": Column(pa.String, nullable=True),
        "AS": Column(pa.String, nullable=True),
        "HST": Column(pa.String, nullable=True),
        "AST": Column(pa.String, nullable=True),
        "HF": Column(pa.String, nullable=True),
        "AF": Column(pa.String, nullable=True),
        "HC": Column(pa.String, nullable=True),
        "AC": Column(pa.String, nullable=True),
        "HY": Column(pa.String, nullable=True),
        "AY": Column(pa.String, nullable=True),
        "HR": Column(pa.String, nullable=True),
        "AR": Column(pa.String, nullable=True),
    },
    coerce=False,  # Don't coerce types for upcoming fixtures
    strict=False,
)

#Pandera's DataFrameSchema provides declarative, testable data contracts, failing fast on missing columns, wrong types, or invalid ranges to prevent silent corruption downstream.

