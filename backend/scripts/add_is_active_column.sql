-- Migration script to add is_active column to teams table
-- This script adds the missing is_active column that the API expects

-- Add the is_active column to the teams table
ALTER TABLE teams 
ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE NOT NULL;

-- Update existing teams to be active by default
UPDATE teams SET is_active = TRUE WHERE is_active IS NULL;

-- Create index on is_active for better query performance
CREATE INDEX IF NOT EXISTS idx_teams_is_active ON teams(is_active);

-- Verify the column was added
\d teams;

SELECT 'Migration completed successfully: is_active column added to teams table' AS status;