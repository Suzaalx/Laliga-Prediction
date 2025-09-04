# LaLiga Predictions & Insights Platform

🚀 **Portfolio-ready football prediction platform** showcasing end-to-end ML engineering with Dixon-Coles modeling, modern web stack, and MLOps automation.

## 🎯 Project Overview

A production-ready platform that combines:
- **Rigorous modeling**: Dixon-Coles Poisson with time decay and low-score corrections
- **Modern web stack**: Next.js/TypeScript frontend + FastAPI backend
- **MLOps automation**: Automated retraining, CI/CD, and monitoring
- **Portfolio appeal**: Deployable, interactive, and recruiter-friendly

## 🏗️ Architecture

```
├── backend/              # FastAPI prediction service
│   ├── app/             # Application code
│   │   ├── api/         # API endpoints and routers
│   │   ├── core/        # Configuration and database
│   │   ├── models/      # Database models and schemas
│   │   ├── services/    # Business logic services
│   │   └── utils/       # Utility functions
│   ├── tests/           # Backend test suite
│   └── scripts/         # Database initialization
├── frontend/            # Next.js web application
│   ├── src/            # Source code
│   │   ├── app/        # Next.js app router
│   │   ├── components/ # React components
│   │   └── lib/        # Utility libraries
│   └── public/         # Static assets
├── src/laliga_pipeline/ # Core ML pipeline
│   ├── dc_model.py     # Enhanced Dixon-Coles model
│   ├── features.py     # Feature engineering pipeline
│   ├── backtest.py     # Model evaluation and backtesting
│   └── loaders.py      # Data loading utilities
├── mlops/              # MLOps automation
│   ├── deploy.py       # Model deployment
│   ├── model_registry.py # Model versioning
│   └── monitoring.py   # Performance monitoring
├── k8s/                # Kubernetes deployment configs
├── data/               # Raw match data (CSV files)
├── artifacts/          # Generated model artifacts
├── tests/              # Core pipeline tests
└── .github/workflows/  # CI/CD automation
```

## 🧠 Model Features

### Dixon-Coles Framework
- **Core intensities**: λ_home = exp(μ + α_home - δ_away + γ), λ_away = exp(μ + α_away - δ_home)
- **Time decay**: Exponential weighting for recent match importance
- **Low-score correlation**: Adjustments for 0-0, 1-0, 0-1, 1-1 outcomes
- **Home advantage**: Dedicated parameter for venue effects

### Feature Engineering
- **Rolling aggregates**: Per-90 stats by team/venue/referee
- **Performance metrics**: Shots, shots on target, corners, cards, fouls
- **Referee effects**: Discipline tendencies and card patterns
- **Form indicators**: Recent performance and momentum

## 🛠️ Tech Stack

### Frontend
- **Next.js 14** with TypeScript and SSR/SSG
- **React** with modern hooks and state management
- **Tailwind CSS** for responsive, modern UI
- **Chart.js/D3** for interactive visualizations

### Backend
- **FastAPI** with async endpoints
- **PostgreSQL** for structured data storage
- **Redis** for caching and session management
- **Pydantic** for data validation

### MLOps
- **GitHub Actions** for CI/CD automation
- **Docker** containerization
- **Model registry** for versioning
- **Monitoring** for drift detection

## 📊 Evaluation Metrics

- **Calibration curves**: Probability reliability assessment
- **Brier score**: Probabilistic accuracy measurement
- **Log loss**: Information-theoretic evaluation
- **Rolling backtests**: Time-series cross-validation

## 📚 Documentation

- **[ML Pipeline Documentation](docs/ML_PIPELINE.md)** - Comprehensive guide to the Dixon-Coles model and feature engineering
- **[API Documentation](docs/API_DOCUMENTATION.md)** - Complete API reference with examples
- **[Kubernetes Deployment](k8s/README.md)** - Container orchestration and deployment guide

## 🚀 How to Run This Project

### Prerequisites
Make sure you have the following installed:
- **Node.js 18+** - [Download here](https://nodejs.org/)
- **Python 3.9+** - [Download here](https://www.python.org/downloads/)
- **Git** - [Download here](https://git-scm.com/downloads)
- **PostgreSQL 14+** (optional for full functionality) - [Download here](https://www.postgresql.org/download/)
- **Docker** (optional for containerized deployment) - [Download here](https://www.docker.com/get-started)

### 🎯 Quick Start (Recommended)

Follow these steps to get the project running locally:

#### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/laliga_pipeline.git
cd laliga_pipeline
```

#### 2. Set Up Python Backend
```bash
# Navigate to backend directory
cd backend

# Create and activate virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### 3. Set Up Frontend (Open New Terminal)
```bash
# Navigate to frontend directory
cd frontend

# Install Node.js dependencies
npm install

# Start the development server
npm run dev
```

#### 4. Access the Application
Once both servers are running:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### 🐳 Alternative: Docker Setup

If you prefer using Docker:

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/laliga_pipeline.git
cd laliga_pipeline

# Build and start all services
docker-compose up -d

# Access the application
# Frontend: http://localhost:3000
# Backend: http://localhost:8000
```

### 🔧 Troubleshooting

#### Common Issues and Solutions

**Port Already in Use:**
```bash
# If port 3000 or 8000 is already in use, kill the process:
# On macOS/Linux:
lsof -ti:3000 | xargs kill -9
lsof -ti:8000 | xargs kill -9

# On Windows:
netstat -ano | findstr :3000
taskkill /PID <PID> /F
```

**Python Virtual Environment Issues:**
```bash
# If virtual environment creation fails:
pip install --upgrade pip
python -m pip install virtualenv
```

**Node.js Dependencies Issues:**
```bash
# Clear npm cache and reinstall:
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

**Database Connection (Optional):**
- The app works with mock data by default
- For full functionality, set up PostgreSQL and update the `.env` file in the backend directory

### 🚀 Advanced Setup Options

#### Option A: Kubernetes Deployment

```bash
# Deploy to Kubernetes cluster
cd k8s
./deploy.sh

# Add to hosts file for local access
echo "127.0.0.1 laliga-predictions.local" >> /etc/hosts

# Access: http://laliga-predictions.local
```

#### Option B: Train Your Own ML Model

```bash
# Install core pipeline
pip install -e .

# Train Dixon-Coles model
python -c "from src.laliga_pipeline.cli import main; main(['train'])"

# Run backtesting
python -c "from src.laliga_pipeline.cli import main; main(['backtest'])"
```

### 📱 What You'll See

Once the application is running, you'll have access to:
- **Homepage**: Interactive dashboard with match predictions
- **Teams**: Detailed team analytics and performance metrics
- **Predictions**: Future match probability predictions
- **Standings**: Current league table with statistics
- **Analytics**: Advanced team and player insights

### 🔄 Development Workflow

```bash
# Make changes to the code
# Backend changes: FastAPI auto-reloads
# Frontend changes: Next.js hot-reloads automatically

# Run tests
pytest tests/ -v                    # Python tests
cd frontend && npm test             # Frontend tests (if available)

# Build for production
cd frontend && npm run build        # Build frontend
cd backend && pip install gunicorn  # Production server
```

## 📈 Features

### Web Interface
- **Fixture predictions**: Pre-match probabilities with confidence intervals
- **Team analytics**: Attack/defense strength visualizations
- **Referee insights**: Card and foul tendency heatmaps
- **Model transparency**: Feature importance and decision explanations

### API Endpoints
- `GET /predictions/{match_id}`: Match probability predictions
- `GET /teams/{team_id}/stats`: Team performance metrics
- `GET /model/info`: Model metadata and performance
- `POST /predictions/batch`: Bulk prediction requests

## 🔄 MLOps Pipeline

### Automated Retraining
- **Nightly jobs**: Retrain with latest match data
- **Performance validation**: Deploy only if metrics improve
- **Rollback capability**: Automatic fallback on failures

### Monitoring
- **Prediction calibration**: Real-time accuracy tracking
- **Feature drift**: Distribution change detection
- **Service health**: API performance and uptime

## 🎯 Portfolio Highlights

✅ **Statistical rigor**: Proven Dixon-Coles framework with modern enhancements
✅ **Production ready**: Containerized, monitored, and scalable
✅ **Modern stack**: 2025-aligned technologies (Next.js, FastAPI, TypeScript)
✅ **MLOps integration**: Automated training, testing, and deployment
✅ **Interactive demo**: Click-through experience for recruiters
✅ **Documentation**: Clear architecture and decision explanations

## 📝 Model Card

### Assumptions
- Goals follow Poisson distributions with team-specific rates
- Recent matches are more predictive (exponential decay)
- Low-score games have correlated outcomes
- Home advantage provides consistent boost

### Limitations
- No injury/transfer modeling
- Weather conditions not considered
- Limited to historical statistical patterns
- Requires minimum training data per team

### Performance
- **Calibration**: Well-calibrated probabilities across score ranges
- **Accuracy**: Competitive with industry benchmarks
- **Robustness**: Stable across different seasons and competitions

## 📁 Project Structure

The project is organized into clear, logical directories:

- **`backend/`** - FastAPI application with comprehensive API endpoints
- **`frontend/`** - Next.js web application with modern UI
- **`src/laliga_pipeline/`** - Core ML pipeline with Dixon-Coles model
- **`mlops/`** - MLOps automation and monitoring
- **`k8s/`** - Kubernetes deployment configurations
- **`docs/`** - Comprehensive documentation
- **`data/`** - Raw match data (CSV files by season)
- **`artifacts/`** - Generated model artifacts and features
- **`tests/`** - Test suites for all components

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run backend tests
cd backend && pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test categories
pytest tests/test_dc_model.py -v  # Model tests
pytest tests/test_features.py -v  # Feature engineering tests
```

## 🔧 Configuration

### Environment Variables

**Backend (.env):**
```bash
DATABASE_URL=postgresql://user:pass@localhost/laliga_predictions
REDIS_URL=redis://localhost:6379
DEBUG=true
ALLOWED_HOSTS=["localhost","127.0.0.1"]
```

**Frontend (.env.local):**
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
NODE_ENV=development
```

### Model Parameters

Key parameters in `src/laliga_pipeline/config.py`:
- **Time decay (xi)**: 0.01 (controls recent match importance)
- **Regularization**: 0.005 (prevents overfitting)
- **Rolling windows**: [5, 10, 20] (form calculation periods)

## 🔮 Future Enhancements

- **xG integration**: Expected goals for improved accuracy
- **Live updates**: Real-time in-match probability updates
- **Player modeling**: Individual player impact assessment
- **Multi-league**: Expand to other European leagues
- **Mobile app**: React Native mobile application
- **Advanced analytics**: Shot maps, pass networks, tactical analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Dixon & Coles (1997) for the foundational model
- Football-Data.co.uk for historical match data
- The open-source community for excellent tools and libraries

---

**Built with ❤️ for the beautiful game and modern ML engineering**