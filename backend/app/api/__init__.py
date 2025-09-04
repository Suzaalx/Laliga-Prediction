from fastapi import APIRouter
from app.api.routes import health, predictions, teams, models, monitoring, matches, data, csv_teams, team_analytics, analytics

api_router = APIRouter()

# Include all route modules
api_router.include_router(
    health.router,
    prefix="/health",
    tags=["health"]
)

api_router.include_router(
    predictions.router,
    prefix="/predictions",
    tags=["predictions"]
)

api_router.include_router(
    teams.router,
    prefix="/teams",
    tags=["teams"]
)

api_router.include_router(
    matches.router,
    prefix="/matches",
    tags=["matches"]
)

api_router.include_router(
    models.router,
    prefix="/models",
    tags=["models"]
)

api_router.include_router(
    monitoring.router,
    prefix="/monitoring",
    tags=["monitoring"]
)

api_router.include_router(
    data.router,
    prefix="/data",
    tags=["data"]
)

api_router.include_router(
    csv_teams.router,
    prefix="/csv-teams",
    tags=["csv-teams"]
)

api_router.include_router(
    team_analytics.router,
    prefix="/team-analytics",
    tags=["team-analytics"]
)

api_router.include_router(
    analytics.router,
    prefix="/analytics",
    tags=["analytics"]
)