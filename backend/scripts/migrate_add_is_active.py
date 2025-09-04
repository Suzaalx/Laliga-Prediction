#!/usr/bin/env python3
"""
Migration script to add is_active column to teams table
"""

import sys
import os
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from sqlalchemy import create_engine, text
from app.core.config import settings

def run_migration():
    """Run the migration to add is_active column"""
    
    # Create database engine
    engine = create_engine(settings.DATABASE_URL)
    
    try:
        with engine.connect() as conn:
            # Start a transaction
            with conn.begin():
                print("Adding is_active column to teams table...")
                
                # Add the is_active column
                conn.execute(text("""
                    ALTER TABLE teams 
                    ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE NOT NULL
                """))
                
                # Update existing teams to be active by default
                conn.execute(text("""
                    UPDATE teams SET is_active = TRUE WHERE is_active IS NULL
                """))
                
                # Create index for better query performance
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_teams_is_active ON teams(is_active)
                """))
                
                print("Migration completed successfully!")
                
                # Verify the column exists
                result = conn.execute(text("""
                    SELECT column_name, data_type, is_nullable, column_default 
                    FROM information_schema.columns 
                    WHERE table_name = 'teams' AND column_name = 'is_active'
                """))
                
                row = result.fetchone()
                if row:
                    print(f"Column verified: {row[0]} ({row[1]}, nullable: {row[2]}, default: {row[3]})")
                else:
                    print("Warning: Could not verify column was added")
                    
    except Exception as e:
        print(f"Migration failed: {e}")
        return False
        
    return True

if __name__ == "__main__":
    success = run_migration()
    sys.exit(0 if success else 1)