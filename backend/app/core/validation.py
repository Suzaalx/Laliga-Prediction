"""Data source validation utilities to ensure exclusive use of specified CSV data."""

from typing import Dict, Any, List, Tuple
from pathlib import Path
import pandas as pd
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("data_validation")

def validate_csv_source(csv_content: str, source_info: Dict[str, Any] = None) -> Tuple[bool, List[str]]:
    """Validate that CSV data comes from approved sources only.
    
    Args:
        csv_content: The CSV content as string
        source_info: Optional metadata about the data source
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    try:
        # Check if data source validation is enabled
        if not settings.VALIDATE_DATA_SOURCE:
            return True, []
            
        # Validate data source type
        if source_info and 'source_type' in source_info:
            source_type = source_info['source_type']
            if source_type not in settings.ALLOWED_DATA_SOURCES:
                errors.append(f"Data source '{source_type}' not in allowed sources: {settings.ALLOWED_DATA_SOURCES}")
        
        # Parse CSV to validate structure
        df = pd.read_csv(pd.io.common.StringIO(csv_content))
        
        # Validate required columns for artifacts data
        required_columns = [
            'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
            'HTHG', 'HTAG', 'HTR', 'Season'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns for artifacts data: {missing_columns}")
            
        # Validate season format (should be YYYY-YY format for artifacts)
        if 'Season' in df.columns:
            invalid_seasons = df[~df['Season'].str.match(r'^\d{4}-\d{2}$', na=False)]
            if not invalid_seasons.empty:
                errors.append(f"Invalid season format found. Expected YYYY-YY format.")
                
        # Validate date range (artifacts should contain historical data)
        if 'Date' in df.columns:
            try:
                dates = pd.to_datetime(df['Date'], errors='coerce')
                min_date = dates.min()
                max_date = dates.max()
                
                # Check for reasonable date range (2000-2025)
                if min_date.year < 2000 or max_date.year > 2025:
                    errors.append(f"Date range {min_date.year}-{max_date.year} outside expected range (2000-2025)")
                    
            except Exception as e:
                errors.append(f"Error validating dates: {str(e)}")
                
        # Log validation results
        if errors:
            logger.warning(f"CSV source validation failed: {errors}")
        else:
            logger.info(f"CSV source validation passed for {len(df)} records")
            
    except Exception as e:
        errors.append(f"Validation error: {str(e)}")
        logger.error(f"CSV validation exception: {str(e)}")
        
    return len(errors) == 0, errors

def validate_artifacts_integrity() -> Tuple[bool, List[str]]:
    """Validate that artifacts directory contains expected files.
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    try:
        artifacts_dir = Path(settings.ARTIFACTS_DIR)
        
        # Check if artifacts directory exists
        if not artifacts_dir.exists():
            errors.append(f"Artifacts directory not found: {artifacts_dir}")
            return False, errors
            
        # Check for required files
        required_files = [
            settings.MATCHES_CSV_FILE,
            'matches_features.csv',
            'team_form.csv'
        ]
        
        for file_name in required_files:
            file_path = artifacts_dir / file_name
            if not file_path.exists():
                errors.append(f"Required artifacts file not found: {file_name}")
            elif file_path.stat().st_size == 0:
                errors.append(f"Artifacts file is empty: {file_name}")
                
        # Validate main matches file
        matches_file = artifacts_dir / settings.MATCHES_CSV_FILE
        if matches_file.exists():
            try:
                df = pd.read_csv(matches_file)
                if len(df) == 0:
                    errors.append(f"Matches file is empty: {settings.MATCHES_CSV_FILE}")
                elif len(df) < 1000:  # Expect at least 1000 matches
                    errors.append(f"Matches file contains too few records: {len(df)}")
                    
            except Exception as e:
                errors.append(f"Error reading matches file: {str(e)}")
                
    except Exception as e:
        errors.append(f"Artifacts validation error: {str(e)}")
        
    return len(errors) == 0, errors

def get_approved_data_source() -> Dict[str, Any]:
    """Get the approved data source configuration.
    
    Returns:
        Dictionary with data source configuration
    """
    return {
        'source_type': settings.DATA_SOURCE_TYPE,
        'artifacts_dir': settings.ARTIFACTS_DIR,
        'matches_file': settings.MATCHES_CSV_FILE,
        'allowed_sources': settings.ALLOWED_DATA_SOURCES,
        'validation_enabled': settings.VALIDATE_DATA_SOURCE
    }