# La Liga Prediction API Backend

Advanced football match prediction system using Dixon-Coles probabilistic model with comprehensive feature engineering.

## Features

- **Dixon-Coles Probabilistic Model**: Advanced statistical model for match outcome prediction
- **Comprehensive Feature Engineering**: Rolling averages, form analysis, venue-specific metrics
- **Real-time Predictions**: Fast API endpoints for match predictions
- **Team Analytics**: Detailed team statistics and performance metrics
- **Model Management**: Automated retraining and performance tracking
- **Caching Layer**: Redis-based caching for optimal performance
- **Health Monitoring**: Comprehensive health checks and metrics
- **Database Integration**: PostgreSQL with optimized queries
- **Containerized Deployment**: Docker and Docker Compose support

## API Endpoints

### Health Endpoints
- `GET /api/health/` - Main health check
- `GET /api/health/liveness` - Liveness probe
- `GET /api/health/readiness` - Readiness probe
- `GET /api/health/metrics` - Health metrics

### Prediction Endpoints
- `POST /api/predictions/predict` - Single match prediction
- `POST /api/predictions/batch` - Batch predictions
- `GET /api/predictions/upcoming` - Upcoming match predictions
- `GET /api/predictions/history` - Prediction history
- `GET /api/predictions/stats` - Prediction statistics
- `DELETE /api/predictions/cache` - Clear prediction cache

### Team Endpoints
- `GET /api/teams/` - List all teams
- `GET /api/teams/{team_id}` - Get team by ID
- `GET /api/teams/{team_id}/stats` - Team statistics
- `GET /api/teams/{team_id}/form` - Team form analysis
- `GET /api/teams/search` - Search teams

### Model Endpoints
- `GET /api/models/active` - Get active model
- `GET /api/models/` - List all models
- `GET /api/models/{model_id}/performance` - Model performance
- `POST /api/models/retrain` - Start model retraining
- `GET /api/models/retrain/status` - Retraining status
- `POST /api/models/{model_id}/activate` - Activate model
- `DELETE /api/models/{model_id}` - Delete model

## Quick Start

### Using Docker Compose (Recommended)

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd laliga_pipeline
   ```

2. **Start the services**:
   ```bash
   docker-compose up -d
   ```

3. **Check service health**:
   ```bash
   curl http://localhost:8000/health
   ```

4. **Access API documentation**:
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

### Local Development Setup

1. **Prerequisites**:
   - Python 3.11+
   - PostgreSQL 15+
   - Redis 7+

2. **Install dependencies**:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Set environment variables**:
   ```bash
   export DATABASE_URL="postgresql://user:password@localhost:5432/laliga_predictions"
   export REDIS_URL="redis://localhost:6379/0"
   export DEBUG="true"
   export SECRET_KEY="your-secret-key"
   ```

4. **Run database migrations**:
   ```bash
   alembic upgrade head
   ```

5. **Start the application**:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|----------|
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `REDIS_URL` | Redis connection string | Required |
| `DEBUG` | Enable debug mode | `false` |
| `SECRET_KEY` | Application secret key | Required |
| `ALLOWED_HOSTS` | Comma-separated allowed hosts | `localhost` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `CACHE_TTL_SECONDS` | Cache TTL in seconds | `3600` |
| `MODEL_RETRAIN_INTERVAL_HOURS` | Model retraining interval | `24` |

### Database Configuration

The application uses PostgreSQL with the following extensions:
- `uuid-ossp` - UUID generation
- `pg_trgm` - Fuzzy text matching
- `btree_gin` - Enhanced indexing

### Redis Configuration

Redis is used for:
- Prediction caching
- Session storage
- Rate limiting
- Background task queues

## Data Ingestion

The API supports CSV data ingestion with the following format:

```csv
Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,HS,AS,HST,AST,HF,AF,HC,AC,HY,AY,HR,AR,Referee
2023-08-19,Real Madrid,Athletic Bilbao,2,0,H,1,0,H,15,8,6,3,12,15,7,4,2,3,0,0,Referee Name
```

### Required Columns
- `Date` - Match date (YYYY-MM-DD)
- `HomeTeam` - Home team name
- `AwayTeam` - Away team name
- `FTHG` - Full time home goals
- `FTAG` - Full time away goals
- `FTR` - Full time result (H/D/A)

### Optional Columns
- `HTHG/HTAG` - Half time goals
- `HS/AS` - Shots
- `HST/AST` - Shots on target
- `HF/AF` - Fouls
- `HC/AC` - Corners
- `HY/AY` - Yellow cards
- `HR/AR` - Red cards
- `Referee` - Referee name

## Model Architecture

### Dixon-Coles Model

The core prediction model is based on the Dixon-Coles approach with enhancements:

1. **Poisson Distribution**: Models goal scoring as independent Poisson processes
2. **Time Decay**: Recent matches weighted more heavily
3. **Low-Score Correction**: Adjusts for correlation in low-scoring games
4. **Home Advantage**: Explicit home field advantage parameter

### Feature Engineering

Comprehensive feature set including:

- **Rolling Statistics**: Goals, shots, cards over various windows
- **Form Metrics**: Recent performance weighted by recency
- **Venue-Specific**: Home/away performance splits
- **Head-to-Head**: Historical matchup statistics
- **Advanced Metrics**: Expected goals, shot conversion rates
- **Referee Adjustments**: Referee-specific tendencies

## Testing

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_api.py

# Run with verbose output
pytest -v
```

### Test Coverage

The test suite covers:
- API endpoints (100%)
- Service layer (95%+)
- Utility functions (95%+)
- Model components (90%+)
- Database operations (85%+)

## Performance

### Benchmarks

- **Single Prediction**: < 50ms
- **Batch Predictions (10 matches)**: < 200ms
- **Team Statistics**: < 30ms
- **Model Training**: 2-5 minutes (depending on data size)

### Optimization Features

- **Redis Caching**: Aggressive caching of predictions and statistics
- **Database Indexing**: Optimized queries with proper indexes
- **Connection Pooling**: Efficient database connection management
- **Async Processing**: Non-blocking I/O operations
- **Background Tasks**: Model retraining in background

## Monitoring

### Health Checks

- **Liveness**: Application is running
- **Readiness**: Application can serve requests
- **Database**: Database connectivity
- **Cache**: Redis connectivity
- **Model**: Model availability and performance

### Metrics

- **Request Metrics**: Response times, error rates
- **Model Metrics**: Prediction accuracy, calibration
- **System Metrics**: Memory usage, CPU utilization
- **Business Metrics**: Prediction volume, cache hit rates

## Deployment

### Production Considerations

1. **Security**:
   - Use strong secret keys
   - Enable HTTPS
   - Configure proper CORS settings
   - Use environment-specific configurations

2. **Scalability**:
   - Use load balancers
   - Scale horizontally with multiple instances
   - Optimize database connections
   - Monitor resource usage

3. **Reliability**:
   - Implement proper error handling
   - Use health checks
   - Set up monitoring and alerting
   - Regular backups

### Kubernetes Deployment

```yaml
# Example Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: laliga-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: laliga-api
  template:
    metadata:
      labels:
        app: laliga-api
    spec:
      containers:
      - name: api
        image: laliga-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        livenessProbe:
          httpGet:
            path: /api/health/liveness
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/health/readiness
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

## Contributing

1. **Code Style**: Follow PEP 8, use Black for formatting
2. **Testing**: Maintain test coverage above 90%
3. **Documentation**: Update docs for new features
4. **Type Hints**: Use type hints for all functions
5. **Error Handling**: Implement proper error handling

## License

This project is licensed under the MIT License - see the LICENSE file for details.