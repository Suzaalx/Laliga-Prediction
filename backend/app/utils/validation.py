import re
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pydantic import BaseModel, validator, ValidationError

from app.core.logging import get_logger

logger = get_logger("validation")

class MatchDataValidator(BaseModel):
    """Pydantic model for match data validation"""
    
    date: datetime
    home_team: str
    away_team: str
    home_goals: Optional[int] = None
    away_goals: Optional[int] = None
    home_goals_ht: Optional[int] = None
    away_goals_ht: Optional[int] = None
    result: Optional[str] = None
    result_ht: Optional[str] = None
    referee: Optional[str] = None
    
    # Shot statistics
    home_shots: Optional[int] = None
    away_shots: Optional[int] = None
    home_shots_target: Optional[int] = None
    away_shots_target: Optional[int] = None
    
    # Fouls and cards
    home_fouls: Optional[int] = None
    away_fouls: Optional[int] = None
    home_yellow_cards: Optional[int] = None
    away_yellow_cards: Optional[int] = None
    home_red_cards: Optional[int] = None
    away_red_cards: Optional[int] = None
    
    # Corners
    home_corners: Optional[int] = None
    away_corners: Optional[int] = None
    
    @validator('home_team', 'away_team')
    def validate_team_names(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Team name cannot be empty')
        if len(v) > 100:
            raise ValueError('Team name too long')
        return v.strip()
    
    @validator('home_goals', 'away_goals', 'home_goals_ht', 'away_goals_ht')
    def validate_goals(cls, v):
        if v is not None and v < 0:
            raise ValueError('Goals cannot be negative')
        if v is not None and v > 20:
            raise ValueError('Goals value seems unrealistic')
        return v
    
    @validator('result', 'result_ht')
    def validate_result(cls, v):
        if v is not None and v not in ['H', 'A', 'D']:
            raise ValueError('Result must be H, A, or D')
        return v
    
    @validator('home_shots', 'away_shots', 'home_shots_target', 'away_shots_target')
    def validate_shots(cls, v):
        if v is not None and v < 0:
            raise ValueError('Shots cannot be negative')
        if v is not None and v > 50:
            raise ValueError('Shots value seems unrealistic')
        return v
    
    @validator('home_fouls', 'away_fouls')
    def validate_fouls(cls, v):
        if v is not None and v < 0:
            raise ValueError('Fouls cannot be negative')
        if v is not None and v > 50:
            raise ValueError('Fouls value seems unrealistic')
        return v
    
    @validator('home_yellow_cards', 'away_yellow_cards')
    def validate_yellow_cards(cls, v):
        if v is not None and v < 0:
            raise ValueError('Yellow cards cannot be negative')
        if v is not None and v > 11:
            raise ValueError('Yellow cards value seems unrealistic')
        return v
    
    @validator('home_red_cards', 'away_red_cards')
    def validate_red_cards(cls, v):
        if v is not None and v < 0:
            raise ValueError('Red cards cannot be negative')
        if v is not None and v > 5:
            raise ValueError('Red cards value seems unrealistic')
        return v
    
    @validator('home_corners', 'away_corners')
    def validate_corners(cls, v):
        if v is not None and v < 0:
            raise ValueError('Corners cannot be negative')
        if v is not None and v > 20:
            raise ValueError('Corners value seems unrealistic')
        return v
    
    @validator('referee')
    def validate_referee(cls, v):
        if v is not None and len(v.strip()) == 0:
            return None
        if v is not None and len(v) > 100:
            raise ValueError('Referee name too long')
        return v.strip() if v else None

class TeamValidator(BaseModel):
    """Pydantic model for team validation"""
    
    name: str
    short_name: Optional[str] = None
    country: Optional[str] = None
    founded: Optional[int] = None
    stadium: Optional[str] = None
    website: Optional[str] = None
    
    @validator('name')
    def validate_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Team name cannot be empty')
        if len(v) > 100:
            raise ValueError('Team name too long')
        return v.strip()
    
    @validator('short_name')
    def validate_short_name(cls, v):
        if v is not None and len(v) > 10:
            raise ValueError('Short name too long')
        return v.strip() if v else None
    
    @validator('founded')
    def validate_founded(cls, v):
        if v is not None and (v < 1800 or v > datetime.now().year):
            raise ValueError('Founded year seems unrealistic')
        return v
    
    @validator('website')
    def validate_website(cls, v):
        if v is not None and v.strip():
            url_pattern = re.compile(
                r'^https?://'  # http:// or https://
                r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+'  # domain...
                r'(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # host...
                r'localhost|'  # localhost...
                r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
                r'(?::\d+)?'  # optional port
                r'(?:/?|[/?]\S+)$', re.IGNORECASE)
            if not url_pattern.match(v.strip()):
                raise ValueError('Invalid website URL')
        return v.strip() if v else None

def validate_match_data(
    match_data: Dict[str, Any],
    strict: bool = True
) -> Tuple[bool, List[str]]:
    """Validate match data dictionary
    
    Args:
        match_data: Dictionary containing match data
        strict: If True, raise validation errors; if False, return warnings
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    
    errors = []
    
    try:
        # Use Pydantic validator
        MatchDataValidator(**match_data)
        
        # Additional business logic validation
        additional_errors = _validate_match_business_logic(match_data)
        errors.extend(additional_errors)
        
    except ValidationError as e:
        for error in e.errors():
            field = '.'.join(str(loc) for loc in error['loc'])
            message = error['msg']
            errors.append(f"{field}: {message}")
    except Exception as e:
        errors.append(f"Unexpected validation error: {str(e)}")
    
    is_valid = len(errors) == 0
    
    if not is_valid and strict:
        logger.error(f"Match data validation failed: {errors}")
    elif errors:
        logger.warning(f"Match data validation warnings: {errors}")
    
    return is_valid, errors

def validate_team_name(
    team_name: str,
    known_teams: Optional[List[str]] = None
) -> Tuple[bool, List[str]]:
    """Validate team name
    
    Args:
        team_name: Team name to validate
        known_teams: List of known team names for fuzzy matching
    
    Returns:
        Tuple of (is_valid, list_of_errors_or_suggestions)
    """
    
    errors = []
    
    try:
        # Basic validation
        if not team_name or len(team_name.strip()) == 0:
            errors.append("Team name cannot be empty")
            return False, errors
        
        if len(team_name) > 100:
            errors.append("Team name too long")
            return False, errors
        
        # Check for invalid characters
        if re.search(r'[<>"\\]', team_name):
            errors.append("Team name contains invalid characters")
            return False, errors
        
        # Fuzzy matching with known teams
        if known_teams:
            from difflib import get_close_matches
            
            clean_name = team_name.strip().lower()
            known_names_lower = [name.lower() for name in known_teams]
            
            if clean_name not in known_names_lower:
                # Find close matches
                close_matches = get_close_matches(
                    clean_name, known_names_lower, n=3, cutoff=0.6
                )
                
                if close_matches:
                    # Map back to original case
                    suggestions = []
                    for match in close_matches:
                        for original in known_teams:
                            if original.lower() == match:
                                suggestions.append(original)
                                break
                    
                    errors.append(f"Unknown team. Did you mean: {', '.join(suggestions)}?")
                else:
                    errors.append("Unknown team name")
        
    except Exception as e:
        errors.append(f"Unexpected validation error: {str(e)}")
    
    is_valid = len(errors) == 0
    return is_valid, errors

def validate_csv_data(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None
) -> Tuple[bool, List[str], pd.DataFrame]:
    """Validate CSV data format and content
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
    
    Returns:
        Tuple of (is_valid, list_of_errors, cleaned_dataframe)
    """
    
    errors = []
    cleaned_df = df.copy()
    
    try:
        # Check if DataFrame is empty
        if df.empty:
            errors.append("CSV file is empty")
            return False, errors, cleaned_df
        
        # Check required columns
        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                errors.append(f"Missing required columns: {missing_columns}")
        
        # Validate each row
        invalid_rows = []
        
        for idx, row in df.iterrows():
            try:
                # Convert row to dict for validation
                row_dict = row.to_dict()
                
                # Handle date conversion
                if 'Date' in row_dict:
                    try:
                        row_dict['date'] = pd.to_datetime(row_dict['Date'], dayfirst=True)
                        del row_dict['Date']
                    except:
                        invalid_rows.append(f"Row {idx + 1}: Invalid date format")
                        continue
                
                # Map column names to expected format
                column_mapping = {
                    'HomeTeam': 'home_team',
                    'AwayTeam': 'away_team',
                    'FTHG': 'home_goals',
                    'FTAG': 'away_goals',
                    'HTHG': 'home_goals_ht',
                    'HTAG': 'away_goals_ht',
                    'FTR': 'result',
                    'HTR': 'result_ht',
                    'Referee': 'referee',
                    'HS': 'home_shots',
                    'AS': 'away_shots',
                    'HST': 'home_shots_target',
                    'AST': 'away_shots_target',
                    'HF': 'home_fouls',
                    'AF': 'away_fouls',
                    'HC': 'home_corners',
                    'AC': 'away_corners',
                    'HY': 'home_yellow_cards',
                    'AY': 'away_yellow_cards',
                    'HR': 'home_red_cards',
                    'AR': 'away_red_cards'
                }
                
                # Apply column mapping
                mapped_row = {}
                for old_col, new_col in column_mapping.items():
                    if old_col in row_dict:
                        mapped_row[new_col] = row_dict[old_col]
                
                # Add unmapped columns
                for col, value in row_dict.items():
                    if col not in column_mapping and col != 'Date':
                        mapped_row[col] = value
                
                # Validate the row
                is_valid, row_errors = validate_match_data(mapped_row, strict=False)
                
                if not is_valid:
                    invalid_rows.extend([f"Row {idx + 1}: {error}" for error in row_errors])
                
            except Exception as e:
                invalid_rows.append(f"Row {idx + 1}: Unexpected error - {str(e)}")
        
        # Add row-level errors
        if invalid_rows:
            errors.extend(invalid_rows[:10])  # Limit to first 10 errors
            if len(invalid_rows) > 10:
                errors.append(f"... and {len(invalid_rows) - 10} more row errors")
        
        # Data quality checks
        quality_errors = _validate_data_quality(df)
        errors.extend(quality_errors)
        
    except Exception as e:
        errors.append(f"Unexpected validation error: {str(e)}")
    
    is_valid = len(errors) == 0
    
    if errors:
        logger.warning(f"CSV validation issues: {errors[:5]}")
    
    return is_valid, errors, cleaned_df

def _validate_match_business_logic(match_data: Dict[str, Any]) -> List[str]:
    """Validate business logic for match data"""
    
    errors = []
    
    try:
        # Check that teams are different
        home_team = match_data.get('home_team', '').strip().lower()
        away_team = match_data.get('away_team', '').strip().lower()
        
        if home_team == away_team:
            errors.append("Home and away teams cannot be the same")
        
        # Check goal consistency
        home_goals = match_data.get('home_goals')
        away_goals = match_data.get('away_goals')
        result = match_data.get('result')
        
        if all(x is not None for x in [home_goals, away_goals, result]):
            expected_result = 'H' if home_goals > away_goals else ('A' if away_goals > home_goals else 'D')
            if result != expected_result:
                errors.append(f"Result '{result}' inconsistent with goals {home_goals}-{away_goals}")
        
        # Check half-time vs full-time goals
        home_goals_ht = match_data.get('home_goals_ht')
        away_goals_ht = match_data.get('away_goals_ht')
        
        if all(x is not None for x in [home_goals, away_goals, home_goals_ht, away_goals_ht]):
            if home_goals_ht > home_goals:
                errors.append("Half-time home goals cannot exceed full-time goals")
            if away_goals_ht > away_goals:
                errors.append("Half-time away goals cannot exceed full-time goals")
        
        # Check shots on target vs total shots
        home_shots = match_data.get('home_shots')
        home_shots_target = match_data.get('home_shots_target')
        away_shots = match_data.get('away_shots')
        away_shots_target = match_data.get('away_shots_target')
        
        if home_shots is not None and home_shots_target is not None:
            if home_shots_target > home_shots:
                errors.append("Home shots on target cannot exceed total shots")
        
        if away_shots is not None and away_shots_target is not None:
            if away_shots_target > away_shots:
                errors.append("Away shots on target cannot exceed total shots")
        
        # Check date reasonableness
        match_date = match_data.get('date')
        if match_date:
            if isinstance(match_date, str):
                try:
                    match_date = pd.to_datetime(match_date)
                except:
                    errors.append("Invalid date format")
                    return errors
            
            if match_date.year < 1900 or match_date.year > datetime.now().year + 1:
                errors.append("Match date seems unrealistic")
    
    except Exception as e:
        errors.append(f"Business logic validation error: {str(e)}")
    
    return errors

def _validate_data_quality(df: pd.DataFrame) -> List[str]:
    """Validate data quality metrics"""
    
    errors = []
    
    try:
        # Check for excessive missing data
        missing_percentage = (df.isnull().sum() / len(df)) * 100
        
        critical_columns = ['Date', 'HomeTeam', 'AwayTeam']
        for col in critical_columns:
            if col in missing_percentage and missing_percentage[col] > 5:
                errors.append(f"High missing data in critical column '{col}': {missing_percentage[col]:.1f}%")
        
        # Check for duplicate matches
        if 'Date' in df.columns and 'HomeTeam' in df.columns and 'AwayTeam' in df.columns:
            duplicates = df.duplicated(subset=['Date', 'HomeTeam', 'AwayTeam'])
            if duplicates.any():
                errors.append(f"Found {duplicates.sum()} duplicate matches")
        
        # Check data distribution
        if 'FTHG' in df.columns and 'FTAG' in df.columns:
            avg_goals = (df['FTHG'].mean() + df['FTAG'].mean()) / 2
            if avg_goals < 0.5 or avg_goals > 5:
                errors.append(f"Unusual average goals per match: {avg_goals:.2f}")
    
    except Exception as e:
        errors.append(f"Data quality validation error: {str(e)}")
    
    return errors

def validate_prediction_request(
    home_team: str,
    away_team: str,
    match_date: Optional[datetime] = None,
    known_teams: Optional[List[str]] = None
) -> Tuple[bool, List[str]]:
    """Validate prediction request parameters
    
    Args:
        home_team: Home team name
        away_team: Away team name
        match_date: Optional match date
        known_teams: List of known team names
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    
    errors = []
    
    # Validate team names
    home_valid, home_errors = validate_team_name(home_team, known_teams)
    away_valid, away_errors = validate_team_name(away_team, known_teams)
    
    if not home_valid:
        errors.extend([f"Home team: {error}" for error in home_errors])
    
    if not away_valid:
        errors.extend([f"Away team: {error}" for error in away_errors])
    
    # Check teams are different
    if home_team.strip().lower() == away_team.strip().lower():
        errors.append("Home and away teams cannot be the same")
    
    # Validate match date if provided
    if match_date:
        if match_date.year < 1900 or match_date.year > datetime.now().year + 2:
            errors.append("Match date seems unrealistic")
    
    is_valid = len(errors) == 0
    return is_valid, errors