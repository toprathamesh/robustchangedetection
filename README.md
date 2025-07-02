# Change Detection System

<img src="https://img.shields.io/badge/version-1.0.0-blue.svg" alt="Version">
<img src="https://img.shields.io/badge/python-3.8%2B-blue.svg" alt="Python Version">
<img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
<img src="https://img.shields.io/badge/status-production--ready-brightgreen.svg" alt="Status">

A robust, automated satellite-based change detection and alerting system for monitoring Areas of Interest (AOIs). This production-grade system distinguishes anthropogenic changes from natural/seasonal variations using state-of-the-art machine learning and temporal analysis.

## 🌟 Key Features

### **Advanced Change Detection**
- **Multi-temporal Analysis**: 3+ year historical baselines with seasonal pattern recognition
- **Siamese CNNs**: Deep learning models specifically designed for change detection
- **Ensemble Methods**: Combines multiple ML approaches for robust predictions
- **Spectral Indices**: 20+ indices including NDVI, NDBI, NDWI, EVI for comprehensive analysis

### **Professional Data Processing**
- **Cloud Masking**: ML-based cloud detection with atmospheric correction
- **Temporal Baselines**: Automatic seasonal baseline construction and anomaly detection
- **Quality Assessment**: Comprehensive data quality metrics and validation
- **Multi-satellite Support**: Sentinel-2, Landsat-8, and Bhoonidhi integration

### **Production Features**
- **Automated Workflows**: Asyncio-based scheduling and monitoring
- **Real-time Alerts**: Email notifications with customizable thresholds
- **GIS Outputs**: Cloud Optimized GeoTIFF, Shapefile, KML/KMZ exports
- **REST API**: Complete Django REST Framework API
- **Interactive Dashboard**: Web-based visualization and monitoring

### **Explainability & Confidence**
- **LIME/SHAP**: Model explainability for transparency
- **Confidence Scoring**: Statistical confidence estimation for all predictions
- **Validation Framework**: Comprehensive metrics (precision, recall, F1-score, IoU)
- **Error Analysis**: Detailed performance analysis and recommendations

## 📁 Project Structure

```
change-detection-system/
├── src/                              # Source code
│   ├── changedetection/              # Main Python package
│   │   ├── core/                     # Core Django app (AOI management, users)
│   │   ├── change_detection/         # ML models and change detection logic
│   │   ├── data_processing/          # Data processing modules
│   │   ├── alerts/                   # Alert system
│   │   └── ml_models/                # Local ML models
│   ├── web/                          # Django project configuration
│   │   └── backend/                  # Django settings, URLs, WSGI/ASGI
│   ├── templates/                    # HTML templates
│   └── manage.py                     # Django management script
├── tests/                            # Test suite
│   ├── unit/                         # Unit tests
│   ├── integration/                  # Integration tests
│   └── fixtures/                     # Test fixtures and sample data
├── docs/                             # Documentation
│   ├── api/                          # API documentation
│   ├── setup/                        # Setup guides
│   └── user_guide/                   # User documentation
├── config/                           # Configuration files
│   ├── requirements.txt              # Python dependencies
│   ├── docker-compose.yml            # Docker configuration
│   ├── Dockerfile                    # Docker image definition
│   └── .env.example                  # Environment variables template
├── pyproject.toml                    # Modern Python project configuration
├── README.md                         # This file
└── LICENSE                           # MIT License
```

## 🚀 Quick Start

### **Prerequisites**

- Python 3.8+
- For production: PostgreSQL with PostGIS extension
- For development: SQLite (included with Python)
- Optional: Redis (for Celery task queue)
- Optional: GDAL/OGR libraries (for advanced GIS features)

### **Quick Local Setup (Recommended for Development)**

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/change-detection-system.git
   cd change-detection-system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -e .[dev]  # Installs package in development mode with dev dependencies
   ```

4. **Set up environment variables**
   ```bash
   # Create .env file for local development
   echo "SECRET_KEY=dev-secret-key-change-in-production" > .env
   echo "DEBUG=True" >> .env
   echo "ALLOWED_HOSTS=localhost,127.0.0.1" >> .env
   echo "USE_SIMPLE_MODELS=True" >> .env
   ```

5. **Set up database and run migrations**
   ```bash
   cd src
   python manage.py makemigrations core change_detection data_processing alerts
   python manage.py migrate
   ```

6. **Create superuser**
   ```bash
   python manage.py createsuperuser
   ```

7. **Start development server**
   ```bash
   python manage.py runserver
   ```

8. **Verify installation**
   ```bash
   cd ..
   python verify_installation.py
   ```

### **Production Setup (PostgreSQL + PostGIS)**

For production deployment with full GIS capabilities:

1. **Install system dependencies**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install postgresql postgresql-contrib postgis gdal-bin
   
   # CentOS/RHEL
   sudo yum install postgresql postgresql-server postgis gdal
   ```

2. **Set up PostgreSQL with PostGIS**
   ```bash
   # Create database with PostGIS extension
   createdb changedetection
   psql changedetection -c "CREATE EXTENSION postgis;"
   ```

3. **Update environment variables**
   ```bash
   # Update .env for production
   echo "DATABASE_URL=postgis://user:password@localhost:5432/changedetection" > .env
   echo "USE_SIMPLE_MODELS=False" >> .env
   echo "DEBUG=False" >> .env
   ```

4. **Run migrations with PostGIS models**
   ```bash
   cd src
   python manage.py migrate
   ```

### **Docker Setup (Recommended for Production)**

1. **Using Docker Compose**
   ```bash
   cp config/.env.example .env
   # Edit .env with your configuration
   
   docker-compose -f config/docker-compose.yml up -d
   ```

2. **Run migrations and create superuser**
   ```bash
   docker-compose exec web python manage.py migrate
   docker-compose exec web python manage.py createsuperuser
   ```

## 🔧 Configuration

### **Environment Variables**

Create a `.env` file from `config/.env.example`:

```env
# Database
DATABASE_URL=postgis://user:password@localhost:5432/changedetection

# Django
SECRET_KEY=your-secret-key
DEBUG=False
ALLOWED_HOSTS=localhost,127.0.0.1,your-domain.com

# Redis/Celery
REDIS_URL=redis://localhost:6379/0

# Email (for alerts)
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_HOST_USER=your-email@gmail.com
EMAIL_HOST_PASSWORD=your-app-password
EMAIL_USE_TLS=True

# Satellite APIs
SENTINEL_HUB_CLIENT_ID=your-sentinel-hub-client-id
SENTINEL_HUB_CLIENT_SECRET=your-client-secret
USGS_USERNAME=your-usgs-username
USGS_PASSWORD=your-usgs-password

# Application Settings
MAX_AOI_SIZE_KM2=1000
CHANGE_THRESHOLD=0.3
ALERT_EMAIL_COOLDOWN_HOURS=24
```

### **Satellite API Setup**

1. **Sentinel Hub**: Register at [Sentinel Hub](https://www.sentinel-hub.com/)
2. **USGS EarthExplorer**: Register at [USGS EarthExplorer](https://earthexplorer.usgs.gov/)
3. **Bhoonidhi**: Contact ISRO for API access

## 📖 Usage

### **Web Interface**

1. Navigate to `http://localhost:8000`
2. Log in with your superuser credentials
3. Create AOIs (Areas of Interest) by drawing polygons on the map
4. Configure change detection parameters
5. Monitor results on the dashboard

### **API Usage**

The system provides a comprehensive REST API. Access the interactive API documentation at `http://localhost:8000/api/docs/`.

#### **Example API Calls**

```python
import requests

# Create AOI
aoi_data = {
    "name": "Test Area",
    "geometry": "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))",
    "is_active": True
}
response = requests.post("http://localhost:8000/api/aois/", json=aoi_data)

# Get change events
response = requests.get("http://localhost:8000/api/change-events/")
change_events = response.json()

# Export change data
export_data = {
    "aoi_id": 1,
    "format": "shapefile",
    "date_from": "2023-01-01",
    "date_to": "2023-12-31"
}
response = requests.post("http://localhost:8000/api/export/", json=export_data)
```

### **Programmatic Usage**

```python
from src.changedetection.data_processing.temporal_analysis import TemporalAnalyzer
from src.changedetection.data_processing.spectral_indices import SpectralIndicesCalculator
from src.changedetection.change_detection.advanced_models import EnsembleChangeDetector

# Initialize components
temporal_analyzer = TemporalAnalyzer(baseline_years=3)
spectral_calc = SpectralIndicesCalculator(platform='sentinel2')
change_detector = EnsembleChangeDetector()

# Process satellite image
indices = spectral_calc.calculate_all_indices(satellite_image)

# Detect changes
results = change_detector.predict_change(before_image, after_image)
```

## 🧪 Testing

### **Run Tests**

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests

# Run with coverage
pytest --cov=src/changedetection --cov-report=html

# Run performance tests
pytest -m slow
```

### **Test Structure**

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test complete workflows and component interactions
- **Fixtures**: Provide consistent test data and mock satellite imagery
- **Performance Tests**: Ensure system performance with large datasets

## 🚢 Deployment

### **Production Deployment**

1. **Server Requirements**
   - 4+ CPU cores
   - 16+ GB RAM
   - 100+ GB storage
   - GPU recommended for ML models

2. **Environment Setup**
   ```bash
   # Install system dependencies
   sudo apt-get update
   sudo apt-get install postgresql-13-postgis-3 redis-server gdal-bin
   
   # Set up production environment
   export DJANGO_SETTINGS_MODULE=web.backend.settings
   export DEBUG=False
   ```

3. **Database Setup**
   ```bash
   # Create production database
   sudo -u postgres createdb changedetection_prod
   sudo -u postgres psql changedetection_prod -c "CREATE EXTENSION postgis;"
   ```

4. **Static Files**
   ```bash
   cd src
   python manage.py collectstatic --noinput
   ```

5. **Process Management**
   ```bash
   # Using systemd for Django
   sudo systemctl enable changedetection-web
   sudo systemctl start changedetection-web
   
   # Using systemd for Celery
   sudo systemctl enable changedetection-worker
   sudo systemctl start changedetection-worker
   ```

### **Docker Production**

```bash
# Build production image
docker build -f config/Dockerfile -t changedetection:latest .

# Deploy with docker-compose
docker-compose -f config/docker-compose.yml -f config/docker-compose.prod.yml up -d
```

## 🔍 Monitoring & Performance

### **System Monitoring**

- **Django Admin**: User management and system overview
- **Celery Flower**: Task monitoring at `http://localhost:5555`
- **API Metrics**: Built-in Django metrics and logging
- **Database Monitoring**: PostgreSQL performance insights

### **Performance Optimization**

- **Database Indexing**: Spatial indexes on geometry fields
- **Caching**: Redis caching for API responses
- **Async Processing**: Celery for background tasks
- **GPU Acceleration**: CUDA support for ML models

### **Scaling**

- **Horizontal Scaling**: Multiple Celery workers
- **Load Balancing**: Nginx/HAProxy for web servers
- **Database Scaling**: Read replicas for PostgreSQL
- **CDN**: Static file distribution

## 🛠️ Development

### **Setting Up Development Environment**

```bash
# Install development dependencies
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install

# Run code formatting
black src/ tests/
isort src/ tests/

# Run linting
flake8 src/ tests/
mypy src/
```

### **Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Run code quality checks (`black`, `isort`, `flake8`)
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

### **Code Quality Standards**

- **100% Test Coverage**: All new code must have tests
- **Type Hints**: Use type annotations for better code clarity
- **Documentation**: Docstrings for all public functions and classes
- **Performance**: Consider performance implications of changes
- **Security**: Follow security best practices for geospatial data

## 📊 System Architecture

### **Core Components**

```mermaid
graph TB
    A[Web Interface] --> B[Django REST API]
    B --> C[Core Models]
    B --> D[Change Detection Engine]
    B --> E[Data Processing Pipeline]
    
    D --> F[Temporal Analysis]
    D --> G[Spectral Indices]
    D --> H[ML Models]
    
    E --> I[Satellite APIs]
    E --> J[Cloud Masking]
    E --> K[GIS Processing]
    
    L[Celery Workers] --> M[Automated Workflows]
    M --> N[Alert System]
    M --> O[Data Export]
    
    P[(PostgreSQL + PostGIS)] --> B
    Q[Redis] --> L
```

### **Data Flow**

1. **AOI Definition**: Users define areas of interest via web interface
2. **Data Acquisition**: Automated fetching from satellite APIs
3. **Preprocessing**: Cloud masking, atmospheric correction, quality assessment
4. **Feature Extraction**: Calculate spectral indices and temporal features
5. **Change Detection**: Apply ML models to detect changes
6. **Validation**: Quality metrics and confidence scoring
7. **Alert Generation**: Notify users of significant changes
8. **Export & Visualization**: Provide results in multiple formats

## 🤝 Support & Community

### **Getting Help**

- **Documentation**: Comprehensive guides in the `docs/` directory
- **API Reference**: Interactive API docs at `/api/docs/`
- **Issues**: Report bugs and request features on GitHub
- **Discussions**: Community discussions on GitHub Discussions

### **Commercial Support**

For commercial support, enterprise features, or custom development:
- Email: support@changedetection.com
- Professional consulting and training available

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **ESA Sentinel Program**: For providing free satellite imagery
- **USGS/NASA Landsat Program**: For long-term earth observation data
- **ISRO Bhoonidhi**: For Indian satellite data access
- **Open Source Community**: For the excellent libraries and tools

## 📈 Roadmap

### **Version 1.1 (Planned)**
- [ ] Real-time processing with Apache Kafka
- [ ] Advanced deep learning models (Vision Transformers)
- [ ] Multi-spectral anomaly detection
- [ ] Mobile application

### **Version 1.2 (Future)**
- [ ] Automatic change type classification improvement
- [ ] Integration with more satellite platforms
- [ ] Advanced statistical analysis tools
- [ ] Machine learning model marketplace

---

**Built with ❤️ for environmental monitoring and sustainable development** 