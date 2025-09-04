"""Advanced Feature Engineering for Football Match Prediction

This module provides comprehensive feature engineering capabilities for football match data,
including rolling statistics, per-90 minute calculations, head-to-head records, momentum
indicators, and venue/referee effects.

Key Features:
- Rolling form calculations with configurable windows
- Per-90 minute normalization for fair comparison
- Head-to-head historical performance
- Momentum and streak detection
- Venue-specific performance metrics
- Referee tendency analysis
- Enhanced long-format data transformation

The pipeline transforms raw match data into ML-ready features that capture:
- Team attacking/defensive strength
- Recent form and momentum
- Historical matchup patterns
- Contextual factors (venue, referee)
"""

import pandas as pd
import numpy as np
from typing import Sequence, Dict, Optional, List
from datetime import datetime, timedelta
import warnings

def to_long_enhanced(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced long format conversion with additional features and per-90 calculations."""
    # Ensure required columns exist
    required_cols = ["Date", "Season", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Home team data
    home_cols = ["Date", "Season", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    stat_cols = ["HS", "HST", "HF", "HC", "HY", "HR"]
    
    # Add optional columns if they exist
    for col in stat_cols:
        if col in df.columns:
            home_cols.append(col)
    
    # Add referee column if exists
    if "Referee" in df.columns:
        home_cols.append("Referee")
    
    home = df[home_cols].copy()
    home["team"] = home["HomeTeam"]
    home["opp"] = home["AwayTeam"]
    home["gf"] = home["FTHG"]
    home["ga"] = home["FTAG"]
    home["venue"] = "home"
    home["home"] = 1
    
    # Map home stats
    home["shots"] = home.get("HS", pd.Series(0, index=home.index)).fillna(0)
    home["sot"] = home.get("HST", pd.Series(0, index=home.index)).fillna(0)
    home["fouls"] = home.get("HF", pd.Series(0, index=home.index)).fillna(0)
    home["corners"] = home.get("HC", pd.Series(0, index=home.index)).fillna(0)
    home["yellows"] = home.get("HY", pd.Series(0, index=home.index)).fillna(0)
    home["reds"] = home.get("HR", pd.Series(0, index=home.index)).fillna(0)
    
    # Away team data
    away_cols = ["Date", "Season", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    away_stat_cols = ["AS", "AST", "AF", "AC", "AY", "AR"]
    
    for col in away_stat_cols:
        if col in df.columns:
            away_cols.append(col)
    
    if "Referee" in df.columns:
        away_cols.append("Referee")
    
    away = df[away_cols].copy()
    away["team"] = away["AwayTeam"]
    away["opp"] = away["HomeTeam"]
    away["gf"] = away["FTAG"]
    away["ga"] = away["FTHG"]
    away["venue"] = "away"
    away["home"] = 0
    
    # Map away stats
    away["shots"] = away.get("AS", pd.Series(0, index=away.index)).fillna(0)
    away["sot"] = away.get("AST", pd.Series(0, index=away.index)).fillna(0)
    away["fouls"] = away.get("AF", pd.Series(0, index=away.index)).fillna(0)
    away["corners"] = away.get("AC", pd.Series(0, index=away.index)).fillna(0)
    away["yellows"] = away.get("AY", pd.Series(0, index=away.index)).fillna(0)
    away["reds"] = away.get("AR", pd.Series(0, index=away.index)).fillna(0)
    
    # Standard columns for concatenation
    base_cols = ["Date", "Season", "team", "opp", "gf", "ga", "shots", "sot", 
                "fouls", "corners", "yellows", "reds", "home", "venue"]
    
    # Add referee if available
    if "Referee" in df.columns:
        base_cols.append("Referee")
    
    long_df = pd.concat([home[base_cols], away[base_cols]], ignore_index=True)
    
    # Calculate derived features
    long_df["shot_accuracy"] = np.where(long_df["shots"] > 0, 
                                       long_df["sot"] / long_df["shots"], 0)
    long_df["goal_conversion"] = np.where(long_df["sot"] > 0, 
                                         long_df["gf"] / long_df["sot"], 0)
    long_df["discipline"] = long_df["yellows"] + 2 * long_df["reds"]
    long_df["goal_difference"] = long_df["gf"] - long_df["ga"]
    
    # Points calculation
    long_df["points"] = np.where(long_df["gf"] > long_df["ga"], 3,
                                np.where(long_df["gf"] == long_df["ga"], 1, 0))
    
    return long_df.sort_values(["Date", "team"])

def calculate_per_90_features(long_df: pd.DataFrame, 
                             match_duration: int = 90) -> pd.DataFrame:
    """Calculate per-90 minute statistics."""
    per_90_cols = ["shots", "sot", "fouls", "corners", "yellows", "reds", "gf", "ga"]
    
    for col in per_90_cols:
        if col in long_df.columns:
            long_df[f"{col}_per90"] = (long_df[col] / match_duration) * 90
    
    return long_df

def rolling_form_enhanced(long_df: pd.DataFrame, 
                         windows: Sequence[int],
                         include_venue: bool = True,
                         include_referee: bool = True,
                         include_opponent: bool = True) -> pd.DataFrame:
    """Enhanced rolling form calculation with venue, referee, and opponent-specific features."""
    long_df = long_df.sort_values(["team", "Date"]).copy()
    
    # Base features to aggregate
    base_features = ["gf", "ga", "shots", "sot", "fouls", "corners", 
                    "yellows", "reds", "shot_accuracy", "goal_conversion", 
                    "discipline", "goal_difference", "points"]
    
    # Add per-90 features if they exist
    per_90_features = [col for col in long_df.columns if col.endswith("_per90")]
    base_features.extend(per_90_features)
    
    # Filter to existing columns
    agg_features = {col: ["mean", "std"] for col in base_features if col in long_df.columns}
    
    all_results = []
    
    for w in windows:
        min_periods = max(3, w // 3)
        
        # Overall rolling features
        overall = (long_df.groupby("team", group_keys=False)
                  .rolling(w, on="Date", min_periods=min_periods)
                  .agg(agg_features)
                  .reset_index())
        
        # Flatten column names
        overall.columns = ["team", "Date"] + [f"{col[0]}_{col[1]}_r{w}" 
                                             for col in overall.columns[2:]]
        
        results = [overall]
        
        # Venue-specific features
        if include_venue and "venue" in long_df.columns:
            for venue in ["home", "away"]:
                venue_df = long_df[long_df["venue"] == venue]
                if len(venue_df) > 0:
                    venue_rolling = (venue_df.groupby("team", group_keys=False)
                                   .rolling(w, on="Date", min_periods=max(1, min_periods//2))
                                   .agg({col: "mean" for col in base_features if col in venue_df.columns})
                                   .reset_index())
                    
                    venue_rolling.columns = (["team", "Date"] + 
                                           [f"{col}_{venue}_r{w}" for col in venue_rolling.columns[2:]])
                    results.append(venue_rolling)
        
        # Referee-specific features (if referee data available)
        if include_referee and "Referee" in long_df.columns:
            referee_stats = (long_df.groupby(["team", "Referee"])
                           .agg({col: "mean" for col in ["fouls", "yellows", "reds"] 
                                if col in long_df.columns})
                           .reset_index())
            
            if len(referee_stats) > 0:
                # Get recent referee tendencies
                recent_ref = (long_df.groupby("Referee")
                            .tail(20)  # Last 20 matches per referee
                            .groupby("Referee")
                            .agg({col: "mean" for col in ["fouls", "yellows", "reds"] 
                                 if col in long_df.columns})
                            .reset_index())
                
                recent_ref.columns = ["Referee"] + [f"ref_{col}_tendency" 
                                                   for col in recent_ref.columns[1:]]
                
                # Merge with main data
                overall = overall.merge(long_df[["team", "Date", "Referee"]], 
                                      on=["team", "Date"], how="left")
                overall = overall.merge(recent_ref, on="Referee", how="left")
        
        # Combine all results for this window
        window_result = results[0]
        for result in results[1:]:
            window_result = window_result.merge(result, on=["team", "Date"], how="outer")
        
        all_results.append(window_result)
    
    # Combine all windows
    final_result = all_results[0]
    for result in all_results[1:]:
        final_result = final_result.merge(result, on=["team", "Date"], how="outer")
    
    return final_result

def calculate_head_to_head_features(long_df: pd.DataFrame, 
                                   windows: Sequence[int] = [5, 10]) -> pd.DataFrame:
    """Calculate head-to-head historical performance features."""
    h2h_features = []
    
    for _, row in long_df.iterrows():
        team = row["team"]
        opp = row["opp"]
        date = row["Date"]
        
        # Get historical matches between these teams
        historical = long_df[
            (long_df["team"] == team) & 
            (long_df["opp"] == opp) & 
            (long_df["Date"] < date)
        ].sort_values("Date", ascending=False)
        
        h2h_row = {"team": team, "Date": date}
        
        for w in windows:
            recent_h2h = historical.head(w)
            
            if len(recent_h2h) > 0:
                h2h_row[f"h2h_gf_mean_{w}"] = recent_h2h["gf"].mean()
                h2h_row[f"h2h_ga_mean_{w}"] = recent_h2h["ga"].mean()
                h2h_row[f"h2h_points_mean_{w}"] = recent_h2h["points"].mean()
                h2h_row[f"h2h_wins_{w}"] = (recent_h2h["points"] == 3).sum()
                h2h_row[f"h2h_matches_{w}"] = len(recent_h2h)
            else:
                h2h_row[f"h2h_gf_mean_{w}"] = np.nan
                h2h_row[f"h2h_ga_mean_{w}"] = np.nan
                h2h_row[f"h2h_points_mean_{w}"] = np.nan
                h2h_row[f"h2h_wins_{w}"] = 0
                h2h_row[f"h2h_matches_{w}"] = 0
        
        h2h_features.append(h2h_row)
    
    return pd.DataFrame(h2h_features)

def calculate_momentum_features(long_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate momentum and streak features."""
    momentum_df = long_df.copy()
    
    # Sort by team and date
    momentum_df = momentum_df.sort_values(["team", "Date"])
    
    # Calculate streaks
    momentum_df["win"] = (momentum_df["points"] == 3).astype(int)
    momentum_df["loss"] = (momentum_df["points"] == 0).astype(int)
    
    # Win/loss streaks
    momentum_df["win_streak"] = (momentum_df.groupby("team")["win"]
                                .apply(lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)))
    
    momentum_df["loss_streak"] = (momentum_df.groupby("team")["loss"]
                                 .apply(lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)))
    
    # Recent form (last 5 matches)
    momentum_df["recent_points"] = (momentum_df.groupby("team")["points"]
                                   .rolling(5, min_periods=1).sum())
    
    momentum_df["recent_gd"] = (momentum_df.groupby("team")["goal_difference"]
                               .rolling(5, min_periods=1).sum())
    
    return momentum_df[["team", "Date", "win_streak", "loss_streak", 
                      "recent_points", "recent_gd"]]

def assemble_enhanced(df: pd.DataFrame, 
                     form: pd.DataFrame,
                     h2h: Optional[pd.DataFrame] = None,
                     momentum: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Enhanced assembly with additional feature sets."""
    # Rename form features for home/away
    fh = form.rename(columns={c: f"H_{c}" for c in form.columns 
                             if c not in ("team", "Date")})
    fa = form.rename(columns={c: f"A_{c}" for c in form.columns 
                             if c not in ("team", "Date")})
    
    # Merge home team features
    m = df.merge(fh, left_on=["HomeTeam", "Date"], 
                right_on=["team", "Date"], how="left")
    
    # Merge away team features
    m = m.merge(fa, left_on=["AwayTeam", "Date"], 
               right_on=["team", "Date"], how="left", suffixes=("", "_away"))
    
    # Clean up duplicate team columns
    m = m.drop(columns=[c for c in ["team_x", "team_y", "team"] if c in m.columns])
    
    # Add head-to-head features if provided
    if h2h is not None:
        h2h_home = h2h.rename(columns={c: f"H_{c}" for c in h2h.columns 
                                      if c not in ("team", "Date")})
        h2h_away = h2h.rename(columns={c: f"A_{c}" for c in h2h.columns 
                                      if c not in ("team", "Date")})
        
        m = m.merge(h2h_home, left_on=["HomeTeam", "Date"], 
                   right_on=["team", "Date"], how="left")
        m = m.merge(h2h_away, left_on=["AwayTeam", "Date"], 
                   right_on=["team", "Date"], how="left", suffixes=("", "_away"))
        
        m = m.drop(columns=[c for c in ["team_x", "team_y", "team"] if c in m.columns])
    
    # Add momentum features if provided
    if momentum is not None:
        mom_home = momentum.rename(columns={c: f"H_{c}" for c in momentum.columns 
                                          if c not in ("team", "Date")})
        mom_away = momentum.rename(columns={c: f"A_{c}" for c in momentum.columns 
                                          if c not in ("team", "Date")})
        
        m = m.merge(mom_home, left_on=["HomeTeam", "Date"], 
                   right_on=["team", "Date"], how="left")
        m = m.merge(mom_away, left_on=["AwayTeam", "Date"], 
                   right_on=["team", "Date"], how="left", suffixes=("", "_away"))
        
        m = m.drop(columns=[c for c in ["team_x", "team_y", "team"] if c in m.columns])
    
    # Create target variables
    if "FTR" in m.columns:
        m["home_win"] = (m["FTR"] == "H").astype(int)
        m["draw"] = (m["FTR"] == "D").astype(int)
        m["away_win"] = (m["FTR"] == "A").astype(int)
    
    # Calculate feature differences (home - away)
    feature_cols = [col for col in m.columns if col.startswith("H_") and 
                   col.replace("H_", "A_") in m.columns]
    
    for col in feature_cols:
        away_col = col.replace("H_", "A_")
        diff_col = col.replace("H_", "diff_")
        m[diff_col] = m[col] - m[away_col]
    
    return m

def create_feature_pipeline(df: pd.DataFrame, 
                           windows: Sequence[int] = [5, 10, 20],
                           include_h2h: bool = True,
                           include_momentum: bool = True,
                           include_venue: bool = True,
                           include_referee: bool = True) -> pd.DataFrame:
    """Complete feature engineering pipeline."""
    print("Starting feature engineering pipeline...")
    
    # Step 1: Convert to long format with enhancements
    print("Converting to enhanced long format...")
    long_df = to_long_enhanced(df)
    
    # Step 2: Calculate per-90 features
    print("Calculating per-90 features...")
    long_df = calculate_per_90_features(long_df)
    
    # Step 3: Calculate rolling form features
    print("Calculating rolling form features...")
    form_features = rolling_form_enhanced(long_df, windows, 
                                         include_venue=include_venue,
                                         include_referee=include_referee)
    
    # Step 4: Calculate head-to-head features
    h2h_features = None
    if include_h2h:
        print("Calculating head-to-head features...")
        h2h_features = calculate_head_to_head_features(long_df)
    
    # Step 5: Calculate momentum features
    momentum_features = None
    if include_momentum:
        print("Calculating momentum features...")
        momentum_features = calculate_momentum_features(long_df)
    
    # Step 6: Assemble final feature set
    print("Assembling final feature set...")
    final_features = assemble_enhanced(df, form_features, h2h_features, momentum_features)
    
    print(f"Feature engineering complete. Final shape: {final_features.shape}")
    return final_features

# Backward compatibility functions
def to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Original to_long function for backward compatibility."""
    return to_long_enhanced(df)

def rolling_form(long_df: pd.DataFrame, windows: Sequence[int]) -> pd.DataFrame:
    """Original rolling_form function for backward compatibility."""
    return rolling_form_enhanced(long_df, windows, 
                                include_venue=False, 
                                include_referee=False, 
                                include_opponent=False)

def assemble(df: pd.DataFrame, form: pd.DataFrame) -> pd.DataFrame:
    """Original assemble function for backward compatibility."""
    return assemble_enhanced(df, form)

# Enhanced feature engineering provides comprehensive match prediction features
# including per-90 statistics, venue effects, referee tendencies, head-to-head
# history, momentum indicators, and rolling form across multiple time windows.