from pathlib import Path
from typing import Iterable
import pandas as pd
from .data_schema import MatchSchema, UpcomingFixtureSchema
from .names import normalize_team

def _parse_dates(s: pd.Series) -> pd.Series:
    # Modern pandas datetime parsing without deprecated parameters
    return pd.to_datetime(s, errors="coerce", dayfirst=True, format="mixed")

def _infer_season(ts: pd.Timestamp) -> str:
    year = ts.year
    start = year if ts.month >= 8 else year - 1
    return f"{start}-{(start+1)%100:02d}"

def load_all(data_dir: Path) -> pd.DataFrame:
    # Exclude upcoming fixture files and incomplete season files from historical data loading
    all_files = sorted(data_dir.glob("*.csv"))
    files = [f for f in all_files if not f.name.startswith("upcoming-") and f.name != "season-2526.csv"]
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir.resolve()}. Expected *.csv files.")
    files: Iterable[Path] = files
    parts = []
    for f in files:
        df = pd.read_csv(f)
        df = MatchSchema.validate(df)  # validate structure
        df["Date"] = _parse_dates(df["Date"])
        df["HomeTeam"] = df["HomeTeam"].map(normalize_team)
        df["AwayTeam"] = df["AwayTeam"].map(normalize_team)
        df = df.dropna(subset=["Date","HomeTeam","AwayTeam","FTHG","FTAG"])
        df["FTR"] = df["FTR"].fillna(df.apply(lambda r: "H" if r.FTHG>r.FTAG else ("A" if r.FTHG<r.FTAG else "D"), axis=1))
        df["Season"] = df["Date"].apply(_infer_season)
        parts.append(df)
    all_df = pd.concat(parts, ignore_index=True).sort_values("Date").reset_index(drop=True)
    return all_df

def load_upcoming_fixtures(data_dir: Path) -> pd.DataFrame:
    """Load upcoming fixtures with empty results for predictions"""
    upcoming_files = sorted(data_dir.glob("upcoming-*.csv"))
    if not upcoming_files:
        return pd.DataFrame()  # Return empty DataFrame if no upcoming fixtures
    
    parts = []
    for f in upcoming_files:
        df = pd.read_csv(f)
        df = UpcomingFixtureSchema.validate(df)  # validate structure for upcoming fixtures
        df["Date"] = _parse_dates(df["Date"])
        df["HomeTeam"] = df["HomeTeam"].map(normalize_team)
        df["AwayTeam"] = df["AwayTeam"].map(normalize_team)
        # Don't drop rows with null goals for upcoming fixtures
        df = df.dropna(subset=["Date","HomeTeam","AwayTeam"])
        df["Season"] = df["Date"].apply(_infer_season)
        parts.append(df)
    
    if parts:
        upcoming_df = pd.concat(parts, ignore_index=True).sort_values("Date").reset_index(drop=True)
        return upcoming_df
    else:
        return pd.DataFrame()

#The loader validates each season, derives missing FTR from goals, infers season boundaries (Aug cutoff), and returns a single, globally sorted timeline for modeling and backtests.

