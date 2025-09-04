-- La Liga Predictions Database Initialization Script
-- This script sets up the initial database structure and configurations

-- Create database if it doesn't exist (handled by docker-compose)
-- CREATE DATABASE IF NOT EXISTS laliga_predictions;

-- Connect to the database
\c laliga_predictions;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For fuzzy text matching
CREATE EXTENSION IF NOT EXISTS "btree_gin";  -- For better indexing

-- Create custom types
CREATE TYPE match_result AS ENUM ('home_win', 'draw', 'away_win');
CREATE TYPE model_status AS ENUM ('training', 'active', 'inactive', 'failed');
CREATE TYPE prediction_status AS ENUM ('pending', 'completed', 'failed');

-- Create sequences for custom IDs
CREATE SEQUENCE IF NOT EXISTS team_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS match_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS model_id_seq START 1;

-- Grant permissions to the application user
GRANT ALL PRIVILEGES ON DATABASE laliga_predictions TO laliga_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO laliga_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO laliga_user;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO laliga_user;

-- Set default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO laliga_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO laliga_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON FUNCTIONS TO laliga_user;

-- Create indexes for common queries (will be created by SQLAlchemy, but good to have as backup)
-- These will be created after tables are set up by the application

-- Performance tuning settings
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;

-- Create a function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create a function for fuzzy team name matching
CREATE OR REPLACE FUNCTION fuzzy_match_team_name(input_name TEXT, threshold REAL DEFAULT 0.6)
RETURNS TABLE(team_id INTEGER, name TEXT, similarity_score REAL) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        t.id,
        t.name,
        similarity(t.name, input_name) as sim_score
    FROM teams t
    WHERE similarity(t.name, input_name) > threshold
    ORDER BY sim_score DESC
    LIMIT 5;
END;
$$ LANGUAGE plpgsql;

-- Create a function to calculate team form
CREATE OR REPLACE FUNCTION calculate_team_form(team_id_param INTEGER, num_matches INTEGER DEFAULT 5)
RETURNS REAL AS $$
DECLARE
    form_points REAL := 0;
    match_record RECORD;
    match_count INTEGER := 0;
BEGIN
    FOR match_record IN
        SELECT 
            CASE 
                WHEN home_team_id = team_id_param THEN
                    CASE 
                        WHEN home_score > away_score THEN 3
                        WHEN home_score = away_score THEN 1
                        ELSE 0
                    END
                WHEN away_team_id = team_id_param THEN
                    CASE 
                        WHEN away_score > home_score THEN 3
                        WHEN away_score = home_score THEN 1
                        ELSE 0
                    END
            END as points
        FROM matches 
        WHERE (home_team_id = team_id_param OR away_team_id = team_id_param)
            AND home_score IS NOT NULL 
            AND away_score IS NOT NULL
        ORDER BY match_date DESC
        LIMIT num_matches
    LOOP
        form_points := form_points + match_record.points;
        match_count := match_count + 1;
    END LOOP;
    
    IF match_count = 0 THEN
        RETURN 0;
    END IF;
    
    RETURN form_points / (match_count * 3.0);  -- Normalize to 0-1 scale
END;
$$ LANGUAGE plpgsql;

-- Create a function to get team statistics
CREATE OR REPLACE FUNCTION get_team_stats(team_id_param INTEGER, season_param TEXT DEFAULT NULL)
RETURNS TABLE(
    matches_played INTEGER,
    wins INTEGER,
    draws INTEGER,
    losses INTEGER,
    goals_for INTEGER,
    goals_against INTEGER,
    goal_difference INTEGER,
    points INTEGER,
    win_rate REAL,
    avg_goals_for REAL,
    avg_goals_against REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*)::INTEGER as matches_played,
        SUM(CASE 
            WHEN (home_team_id = team_id_param AND home_score > away_score) OR 
                 (away_team_id = team_id_param AND away_score > home_score) THEN 1 
            ELSE 0 
        END)::INTEGER as wins,
        SUM(CASE 
            WHEN home_score = away_score THEN 1 
            ELSE 0 
        END)::INTEGER as draws,
        SUM(CASE 
            WHEN (home_team_id = team_id_param AND home_score < away_score) OR 
                 (away_team_id = team_id_param AND away_score < home_score) THEN 1 
            ELSE 0 
        END)::INTEGER as losses,
        SUM(CASE 
            WHEN home_team_id = team_id_param THEN home_score 
            ELSE away_score 
        END)::INTEGER as goals_for,
        SUM(CASE 
            WHEN home_team_id = team_id_param THEN away_score 
            ELSE home_score 
        END)::INTEGER as goals_against,
        (SUM(CASE 
            WHEN home_team_id = team_id_param THEN home_score 
            ELSE away_score 
        END) - SUM(CASE 
            WHEN home_team_id = team_id_param THEN away_score 
            ELSE home_score 
        END))::INTEGER as goal_difference,
        (SUM(CASE 
            WHEN (home_team_id = team_id_param AND home_score > away_score) OR 
                 (away_team_id = team_id_param AND away_score > home_score) THEN 3
            WHEN home_score = away_score THEN 1 
            ELSE 0 
        END))::INTEGER as points,
        (SUM(CASE 
            WHEN (home_team_id = team_id_param AND home_score > away_score) OR 
                 (away_team_id = team_id_param AND away_score > home_score) THEN 1 
            ELSE 0 
        END)::REAL / NULLIF(COUNT(*), 0)) as win_rate,
        (SUM(CASE 
            WHEN home_team_id = team_id_param THEN home_score 
            ELSE away_score 
        END)::REAL / NULLIF(COUNT(*), 0)) as avg_goals_for,
        (SUM(CASE 
            WHEN home_team_id = team_id_param THEN away_score 
            ELSE home_score 
        END)::REAL / NULLIF(COUNT(*), 0)) as avg_goals_against
    FROM matches 
    WHERE (home_team_id = team_id_param OR away_team_id = team_id_param)
        AND home_score IS NOT NULL 
        AND away_score IS NOT NULL
        AND (season_param IS NULL OR season = season_param);
END;
$$ LANGUAGE plpgsql;

-- Create indexes for performance (these will also be created by SQLAlchemy)
CREATE INDEX IF NOT EXISTS idx_matches_teams ON matches(home_team_id, away_team_id);
CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date);
CREATE INDEX IF NOT EXISTS idx_matches_season ON matches(season);
CREATE INDEX IF NOT EXISTS idx_teams_name_trgm ON teams USING gin(name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_predictions_match ON predictions(match_id);
CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at);

-- Insert some initial data if needed
-- This will be handled by the application's data ingestion process

-- Log the completion
DO $$
BEGIN
    RAISE NOTICE 'La Liga Predictions database initialization completed successfully';
END $$;