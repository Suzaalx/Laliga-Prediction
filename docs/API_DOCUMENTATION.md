# API Documentation

## Overview

The La Liga Prediction API is a FastAPI-based service that provides football match predictions, team analytics, and model insights. The API is designed for production use with comprehensive error handling, caching, and monitoring.

## Base URL

- **Development**: `http://localhost:8000`
- **Production**: `https://your-domain.com`

## Authentication

Currently, the API does not require authentication for public endpoints. Future versions may include API key authentication for rate limiting and usage tracking.

## API Endpoints

### Root Endpoints

#### GET `/`
Returns API information and available endpoints.

**Response:**
```json
{
  "message": "La Liga Prediction API",
  "version": "1.0.0",
  "description": "Advanced football match prediction system",
  "endpoints": {
    "predictions": "/api/predictions",
    "teams": "/api/teams",
    "model": "/api/model",
    "health": "/health",
    "docs": "/docs"
  },
  "features": [
    "Dixon-Coles model predictions",
    "Team performance analytics",
    "Real-time monitoring",
    "Comprehensive feature engineering"
  ]
}
```

#### GET `/health`
Simple health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-02T10:30:00Z"
}
```

### Prediction Endpoints

#### GET `/api/predictions/match/{home_team}/{away_team}`
Get prediction for a specific match between two teams.

**Parameters:**
- `home_team` (string): Home team name
- `away_team` (string): Away team name
- `date` (optional, query): Match date (YYYY-MM-DD format)

**Response:**
```json
{
  "match": {
    "home_team": "Real Madrid",
    "away_team": "Barcelona",
    "date": "2025-01-15",
    "venue": "Santiago Bernabéu"
  },
  "predictions": {
    "home_win": 0.45,
    "draw": 0.28,
    "away_win": 0.27
  },
  "expected_goals": {
    "home": 1.8,
    "away": 1.3
  },
  "most_likely_scores": [
    {"home": 1, "away": 0, "probability": 0.18},
    {"home": 2, "away": 1, "probability": 0.15},
    {"home": 1, "away": 1, "probability": 0.14}
  ],
  "confidence": {
    "level": "high",
    "score": 0.82,
    "factors": ["sufficient_data", "recent_form", "head_to_head"]
  },
  "model_info": {
    "version": "1.0.0",
    "last_trained": "2025-01-01T00:00:00Z",
    "features_used": 45
  }
}
```

#### POST `/api/predictions/batch`
Get predictions for multiple matches.

**Request Body:**
```json
{
  "matches": [
    {
      "home_team": "Real Madrid",
      "away_team": "Barcelona",
      "date": "2025-01-15"
    },
    {
      "home_team": "Atletico Madrid",
      "away_team": "Sevilla",
      "date": "2025-01-16"
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "match_id": 0,
      "match": {...},
      "predictions": {...},
      "expected_goals": {...}
    },
    {
      "match_id": 1,
      "match": {...},
      "predictions": {...},
      "expected_goals": {...}
    }
  ],
  "summary": {
    "total_matches": 2,
    "processed": 2,
    "errors": 0
  }
}
```

#### GET `/api/predictions/upcoming`
Get predictions for upcoming fixtures.

**Query Parameters:**
- `days` (optional, default=7): Number of days ahead to fetch
- `team` (optional): Filter by specific team
- `limit` (optional, default=20): Maximum number of matches

**Response:**
```json
{
  "fixtures": [
    {
      "match_id": "laliga_2025_001",
      "date": "2025-01-15",
      "home_team": "Real Madrid",
      "away_team": "Barcelona",
      "predictions": {...},
      "expected_goals": {...}
    }
  ],
  "metadata": {
    "total_fixtures": 10,
    "date_range": {
      "start": "2025-01-15",
      "end": "2025-01-22"
    }
  }
}
```

### Team Endpoints

#### GET `/api/teams/{team_name}/stats`
Get comprehensive statistics for a specific team.

**Parameters:**
- `team_name` (string): Team name
- `season` (optional, query): Specific season (e.g., "2024-25")
- `venue` (optional, query): "home", "away", or "all" (default)

**Response:**
```json
{
  "team": {
    "name": "Real Madrid",
    "season": "2024-25",
    "venue": "all"
  },
  "performance": {
    "matches_played": 20,
    "wins": 14,
    "draws": 4,
    "losses": 2,
    "win_percentage": 0.70,
    "points": 46,
    "points_per_game": 2.30
  },
  "attacking": {
    "goals_scored": 42,
    "goals_per_game": 2.10,
    "shots_per_game": 15.2,
    "shots_on_target_per_game": 6.8,
    "conversion_rate": 0.138,
    "big_chances_created": 38
  },
  "defensive": {
    "goals_conceded": 18,
    "goals_conceded_per_game": 0.90,
    "clean_sheets": 8,
    "clean_sheet_percentage": 0.40,
    "shots_conceded_per_game": 9.5,
    "save_percentage": 0.72
  },
  "discipline": {
    "yellow_cards": 35,
    "red_cards": 2,
    "fouls_per_game": 11.2,
    "fouls_conceded_per_game": 10.8
  },
  "form": {
    "last_5_games": "WWDWW",
    "last_10_games": "WWDWWLWWDW",
    "recent_form_rating": 8.2,
    "momentum": "positive"
  },
  "strengths": {
    "attacking_strength": 1.25,
    "defensive_strength": 0.78,
    "home_advantage": 0.32,
    "overall_rating": 85.4
  }
}
```

#### GET `/api/teams/{team_name}/form`
Get recent form and performance trends.

**Response:**
```json
{
  "team": "Real Madrid",
  "recent_matches": [
    {
      "date": "2025-01-10",
      "opponent": "Sevilla",
      "venue": "home",
      "result": "W",
      "score": "3-1",
      "performance_rating": 8.5
    }
  ],
  "form_analysis": {
    "last_5_form": "WWDWW",
    "points_last_5": 13,
    "goals_scored_last_5": 11,
    "goals_conceded_last_5": 3,
    "trend": "improving",
    "momentum_score": 8.2
  },
  "performance_metrics": {
    "attacking_form": 8.5,
    "defensive_form": 7.8,
    "overall_form": 8.1,
    "consistency": 7.9
  }
}
```

#### GET `/api/teams/rankings`
Get current team rankings and power ratings.

**Query Parameters:**
- `metric` (optional): "attacking", "defensive", "overall" (default)
- `venue` (optional): "home", "away", "all" (default)
- `limit` (optional, default=20): Number of teams to return

**Response:**
```json
{
  "rankings": [
    {
      "rank": 1,
      "team": "Real Madrid",
      "rating": 85.4,
      "attacking_strength": 1.25,
      "defensive_strength": 0.78,
      "recent_form": 8.2
    }
  ],
  "metadata": {
    "metric": "overall",
    "venue": "all",
    "last_updated": "2025-01-02T10:00:00Z",
    "total_teams": 20
  }
}
```

### Model Endpoints

#### GET `/api/model/info`
Get model metadata and performance information.

**Response:**
```json
{
  "model": {
    "name": "Enhanced Dixon-Coles",
    "version": "1.0.0",
    "type": "bivariate_poisson",
    "last_trained": "2025-01-01T00:00:00Z",
    "training_data_size": 5420,
    "features_count": 45
  },
  "performance": {
    "brier_score": 0.234,
    "log_loss": 0.678,
    "calibration_score": 0.92,
    "accuracy_home_win": 0.58,
    "accuracy_draw": 0.31,
    "accuracy_away_win": 0.55
  },
  "parameters": {
    "time_decay": 0.01,
    "home_advantage": 0.25,
    "low_score_correlation": -0.12,
    "regularization": 0.005
  },
  "features": {
    "categories": [
      "team_strength",
      "recent_form",
      "head_to_head",
      "venue_effects",
      "referee_tendencies"
    ],
    "most_important": [
      "attacking_strength_home",
      "defensive_strength_away",
      "recent_form_5_games",
      "home_advantage",
      "head_to_head_last_10"
    ]
  }
}
```

#### GET `/api/model/calibration`
Get model calibration analysis.

**Response:**
```json
{
  "calibration": {
    "overall_score": 0.92,
    "bins": [
      {
        "probability_range": [0.0, 0.1],
        "predicted_probability": 0.05,
        "observed_frequency": 0.048,
        "count": 245,
        "calibration_error": 0.002
      }
    ],
    "reliability_curve": {
      "x_values": [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
      "y_values": [0.048, 0.152, 0.248, 0.351, 0.447, 0.553, 0.648, 0.751, 0.849, 0.952]
    }
  },
  "analysis": {
    "is_well_calibrated": true,
    "overconfidence_bias": 0.002,
    "underconfidence_bias": -0.001,
    "sharpness": 0.78
  }
}
```

### Monitoring Endpoints

#### GET `/metrics`
Get application metrics for monitoring.

**Response:**
```json
{
  "api": {
    "requests_total": 15420,
    "requests_per_minute": 12.5,
    "average_response_time": 0.145,
    "error_rate": 0.002
  },
  "model": {
    "predictions_generated": 8934,
    "cache_hit_rate": 0.78,
    "average_prediction_time": 0.023
  },
  "system": {
    "uptime_seconds": 86400,
    "memory_usage_mb": 512,
    "cpu_usage_percent": 15.2
  }
}
```

#### GET `/monitoring/dashboard`
Get comprehensive monitoring dashboard data.

**Query Parameters:**
- `days` (optional, default=7): Number of days of data to include

**Response:**
```json
{
  "timeframe": {
    "start_date": "2024-12-26",
    "end_date": "2025-01-02",
    "days": 7
  },
  "prediction_quality": {
    "total_predictions": 1250,
    "calibration_score": 0.91,
    "brier_score": 0.236,
    "accuracy_trend": "stable"
  },
  "api_performance": {
    "total_requests": 5420,
    "average_response_time": 0.152,
    "error_rate": 0.001,
    "uptime_percentage": 99.98
  },
  "feature_drift": {
    "detected_drift": false,
    "drift_score": 0.023,
    "threshold": 0.1,
    "last_check": "2025-01-02T09:00:00Z"
  }
}
```

## Error Handling

### Error Response Format

All errors follow a consistent format:

```json
{
  "error": {
    "code": "TEAM_NOT_FOUND",
    "message": "Team 'Invalid Team' not found in database",
    "details": {
      "team_name": "Invalid Team",
      "available_teams": ["Real Madrid", "Barcelona", "..."]
    },
    "timestamp": "2025-01-02T10:30:00Z",
    "request_id": "req_123456789"
  }
}
```

### HTTP Status Codes

- **200 OK**: Successful request
- **400 Bad Request**: Invalid request parameters
- **404 Not Found**: Resource not found (team, match, etc.)
- **422 Unprocessable Entity**: Validation error
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: Server error
- **503 Service Unavailable**: Service temporarily unavailable

### Common Error Codes

- `TEAM_NOT_FOUND`: Specified team not found
- `INVALID_DATE_FORMAT`: Date parameter in wrong format
- `INSUFFICIENT_DATA`: Not enough data for prediction
- `MODEL_NOT_READY`: Model not trained or unavailable
- `VALIDATION_ERROR`: Request validation failed
- `RATE_LIMIT_EXCEEDED`: Too many requests

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Free tier**: 100 requests per hour
- **Authenticated**: 1000 requests per hour
- **Batch endpoints**: Lower limits apply

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1641123600
```

## Caching

The API uses Redis caching for improved performance:

- **Team statistics**: Cached for 1 hour
- **Match predictions**: Cached for 30 minutes
- **Model metadata**: Cached for 24 hours
- **Rankings**: Cached for 2 hours

Cache headers indicate cache status:
```
X-Cache-Status: HIT
X-Cache-TTL: 1800
```

## SDK and Client Libraries

### Python Client Example

```python
import requests

class LaLigaAPI:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def get_prediction(self, home_team, away_team, date=None):
        url = f"{self.base_url}/api/predictions/match/{home_team}/{away_team}"
        params = {"date": date} if date else {}
        response = requests.get(url, params=params)
        return response.json()
    
    def get_team_stats(self, team_name, season=None):
        url = f"{self.base_url}/api/teams/{team_name}/stats"
        params = {"season": season} if season else {}
        response = requests.get(url, params=params)
        return response.json()

# Usage
api = LaLigaAPI()
prediction = api.get_prediction("Real Madrid", "Barcelona")
stats = api.get_team_stats("Real Madrid")
```

### JavaScript Client Example

```javascript
class LaLigaAPI {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async getPrediction(homeTeam, awayTeam, date = null) {
        const url = `${this.baseUrl}/api/predictions/match/${homeTeam}/${awayTeam}`;
        const params = date ? `?date=${date}` : '';
        const response = await fetch(url + params);
        return response.json();
    }
    
    async getTeamStats(teamName, season = null) {
        const url = `${this.baseUrl}/api/teams/${teamName}/stats`;
        const params = season ? `?season=${season}` : '';
        const response = await fetch(url + params);
        return response.json();
    }
}

// Usage
const api = new LaLigaAPI();
const prediction = await api.getPrediction('Real Madrid', 'Barcelona');
const stats = await api.getTeamStats('Real Madrid');
```

## Testing

The API includes comprehensive test coverage:

```bash
# Run backend tests
cd backend
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html

# Run specific test categories
pytest tests/test_api.py -v  # API endpoint tests
pytest tests/test_services.py -v  # Service layer tests
```

## Deployment

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# API will be available at http://localhost:8000
```

### Kubernetes Deployment

```bash
# Deploy to Kubernetes
cd k8s
./deploy.sh

# Access via ingress
echo "127.0.0.1 laliga-predictions.local" >> /etc/hosts
# Visit http://laliga-predictions.local
```

## Support

For API support and questions:

- **Documentation**: `/docs` endpoint (Swagger UI)
- **Health Check**: `/health` endpoint
- **Monitoring**: `/monitoring/dashboard` endpoint
- **Issues**: GitHub repository issues

## Changelog

### v1.0.0 (2025-01-02)
- Initial API release
- Enhanced Dixon-Coles model
- Comprehensive team statistics
- Real-time predictions
- Monitoring and health checks
- Docker and Kubernetes support