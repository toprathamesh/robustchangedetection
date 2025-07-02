# 🛰️ Satellite Change Detection System

<div align="center">

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Django](https://img.shields.io/badge/django-4.2%2B-green.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-production--ready-brightgreen.svg)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)

*A production-grade, AI-powered satellite change detection system for automated monitoring of environmental and anthropogenic changes.*

[🚀 Quick Start](#-quick-start) • [📖 Documentation](#-documentation) • [🎯 Use Cases](#-use-cases) • [🧪 Testing](#-testing) • [🚀 Deploy](#-deployment)

</div>

---

## 📋 Table of Contents

- [🌟 Overview](#-overview)
- [✨ Key Features](#-key-features)
- [🏗️ Architecture](#️-architecture)
- [🚀 Quick Start](#-quick-start)
- [🎯 Use Cases](#-use-cases)
- [📖 Documentation](#-documentation)
- [🧪 Testing](#-testing)
- [🚀 Deployment](#-deployment)
- [🔧 Configuration](#-configuration)
- [💻 CLI Usage](#-cli-usage)
- [🌐 Web Interface](#-web-interface)
- [🔄 End-to-End Workflow](#-end-to-end-workflow)
- [🤝 Contributing](#-contributing)
- [📝 License](#-license)

---

## 🌟 Overview

The **Satellite Change Detection System** is a comprehensive, production-ready platform that automatically monitors Areas of Interest (AOIs) for environmental and anthropogenic changes using satellite imagery. Built with modern Python and Django, it combines advanced machine learning, temporal analysis, and geospatial processing to deliver reliable change detection with confidence scoring and explainability.

### **What makes this system unique:**

🎯 **Intelligent Change Detection**: Distinguishes real changes from seasonal variations using multi-year baselines
🧠 **AI-Powered Analysis**: Siamese CNNs and ensemble methods for robust predictions
📊 **Comprehensive Metrics**: 20+ spectral indices (NDVI, NDBI, NDWI, EVI) for detailed analysis
🌍 **Multi-Satellite Support**: Integrates Sentinel-2, Landsat-8, and Bhoonidhi data sources
⚡ **Real-time Monitoring**: Automated workflows with email alerts and customizable thresholds
🗃️ **Professional Outputs**: GIS-ready exports (Shapefile, GeoTIFF, KML/KMZ)
🔍 **Explainable AI**: LIME/SHAP integration for model transparency

---

## ✨ Key Features

### 🧠 **Advanced Machine Learning**
- **Siamese CNNs** for specialized change detection
- **Ensemble Methods** combining multiple algorithms
- **Confidence Scoring** with statistical validation
- **Model Explainability** using LIME and SHAP
- **Local ML Models** - no cloud dependencies required

### 📊 **Comprehensive Analysis**
- **Temporal Baselines** with 3+ year historical analysis
- **Spectral Indices** - 20+ indices for multi-dimensional analysis
- **Cloud Masking** with ML-based atmospheric correction
- **Quality Metrics** - precision, recall, F1-score, IoU
- **Anomaly Detection** for seasonal pattern recognition

### 🌐 **Production-Ready Platform**
- **REST API** with Django REST Framework
- **Interactive Dashboard** with real-time monitoring
- **Automated Workflows** using asyncio and Celery
- **Email Alerts** with customizable thresholds
- **GIS Integration** with professional export formats

### 🛠️ **Developer-Friendly**
- **Modern Python Package** structure (`src/` layout)
- **Docker Support** for easy deployment
- **Comprehensive Testing** suite
- **CLI Tools** for automation
- **SQLite Development** mode (no PostGIS required for dev)

---

## 🏗️ Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        WEB[Web Dashboard]
        API[REST API]
        CLI[CLI Tools]
    end
    
    subgraph "Application Layer"
        CORE[Core Management]
        DETECTION[Change Detection]
        PROCESSING[Data Processing]
        ALERTS[Alert System]
    end
    
    subgraph "ML & Analysis Layer"
        ML[ML Models]
        SPECTRAL[Spectral Analysis]
        TEMPORAL[Temporal Analysis]
        CONFIDENCE[Confidence Scoring]
    end
    
    subgraph "Data Layer"
        DB[(Database)]
        STORAGE[File Storage]
        CACHE[Redis Cache]
    end
    
    subgraph "External Services"
        SENTINEL[Sentinel Hub]
        LANDSAT[USGS Landsat]
        BHOONIDHI[Bhoonidhi API]
        EMAIL[Email Service]
    end
    
    WEB --> CORE
    API --> CORE
    CLI --> CORE
    
    CORE --> ML
    DETECTION --> ML
    PROCESSING --> SPECTRAL
    PROCESSING --> TEMPORAL
    
    ML --> CONFIDENCE
    SPECTRAL --> CONFIDENCE
    TEMPORAL --> CONFIDENCE
    
    DETECTION --> ALERTS
    ALERTS --> EMAIL
    
    CORE --> DB
    PROCESSING --> STORAGE
    DETECTION --> CACHE
    
    PROCESSING --> SENTINEL
    PROCESSING --> LANDSAT
    PROCESSING --> BHOONIDHI
    
    style ML fill:#e1f5fe
    style CONFIDENCE fill:#f3e5f5
    style ALERTS fill:#fff3e0
```

### **Project Structure**

```
change-detection-system/
├── 📁 src/                          # Source code
│   ├── 📦 changedetection/          # Main Python package
│   │   ├── 🏠 core/                 # User & AOI management
│   │   ├── 🧠 change_detection/     # ML models & detection logic  
│   │   ├── 📊 data_processing/      # Satellite data processing
│   │   ├── 🚨 alerts/               # Alert & notification system
│   │   ├── 🤖 ml_models/            # Local ML model implementations
│   │   └── 🖥️ cli.py                # Command-line interface
│   ├── 🌐 web/                      # Django web application
│   │   └── ⚙️ backend/              # Settings, URLs, WSGI/ASGI
│   ├── 🎨 templates/                # HTML templates
│   └── 🔧 manage.py                 # Django management
├── 🧪 tests/                        # Comprehensive test suite
│   ├── 🔬 unit/                     # Unit tests
│   ├── 🔗 integration/              # Integration tests
│   └── 📊 fixtures/                 # Test data & fixtures
├── 📚 docs/                         # Documentation
├── ⚙️ config/                       # Configuration files
├── 📋 pyproject.toml                # Python project config
├── 📄 README.md                     # This documentation
└── 📜 LICENSE                       # MIT License
```

---

## 🚀 Quick Start

### **Prerequisites**

- 🐍 Python 3.8+
- 🗄️ SQLite (development) or PostgreSQL+PostGIS (production)
- 📦 Git

### **⚡ 30-Second Setup**

```bash
# Clone and setup
git clone https://github.com/your-username/change-detection-system.git
cd change-detection-system

# Install and configure
pip install -e .[dev]
echo "SECRET_KEY=dev-key" > .env
echo "DEBUG=True" >> .env
echo "USE_SIMPLE_MODELS=True" >> .env

# Initialize database
cd src && python manage.py migrate
python manage.py createsuperuser

# Start the system
python manage.py runserver
```

Open [http://localhost:8000](http://localhost:8000) and start detecting changes! 🎉

### **🔍 Verify Installation**

```bash
# Run comprehensive verification
python verify_installation.py

# Expected output:
# ✅ All essential files present
# ✅ Main package import successful
# ✅ Django configuration successful
# 🎉 System ready for production use!
```

---

## 🎯 Use Cases

### 🌳 **Environmental Monitoring**
- **Deforestation Detection**: Monitor forest cover changes
- **Urban Expansion**: Track city growth and land use changes  
- **Agriculture Monitoring**: Crop health and harvest timing
- **Water Body Changes**: Lake, river, and wetland monitoring

### 🏗️ **Infrastructure & Development**
- **Construction Monitoring**: New building and infrastructure development
- **Mining Operations**: Quarry and extraction site monitoring
- **Transportation**: New road and railway development
- **Industrial Expansion**: Factory and industrial area changes

### 🌿 **Conservation & Research**
- **Protected Area Monitoring**: National park and reserve surveillance
- **Habitat Change**: Wildlife habitat modification tracking
- **Climate Impact**: Long-term environmental change assessment
- **Disaster Response**: Post-disaster damage assessment

### 🏭 **Compliance & Enforcement**
- **Regulatory Compliance**: Environmental regulation monitoring
- **Illegal Activity Detection**: Unauthorized construction or clearing
- **Resource Management**: Sustainable development monitoring
- **Impact Assessment**: Environmental impact evaluation

---

## 📖 Documentation

### **🔧 Environment Variables**

Create a `.env` file with these configurations:

```bash
# === Core Django Settings ===
SECRET_KEY=your-secure-secret-key-here
DEBUG=False                          # Set to True for development
ALLOWED_HOSTS=localhost,127.0.0.1,yourdomain.com
USE_SIMPLE_MODELS=True              # False for production with PostGIS

# === Database Configuration ===
# Development (SQLite)
DATABASE_URL=sqlite:///db.sqlite3

# Production (PostgreSQL+PostGIS)  
DATABASE_URL=postgis://user:password@localhost:5432/changedetection

# === Email Settings (for alerts) ===
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_HOST_USER=your-email@gmail.com
EMAIL_HOST_PASSWORD=your-app-password
EMAIL_USE_TLS=True
DEFAULT_FROM_EMAIL=alerts@yourcompany.com

# === Redis/Celery (optional) ===
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0

# === Satellite API Keys (optional) ===
SENTINEL_HUB_CLIENT_ID=your-client-id
SENTINEL_HUB_CLIENT_SECRET=your-client-secret
USGS_USERNAME=your-usgs-username
USGS_PASSWORD=your-usgs-password
BHOONIDHI_API_KEY=your-bhoonidhi-key

# === Application Settings ===
MAX_AOI_SIZE_KM2=1000               # Maximum AOI size in square kilometers
CHANGE_THRESHOLD=0.3                # Default change detection threshold
ALERT_EMAIL_COOLDOWN_HOURS=24       # Minimum hours between alert emails
TEMPORAL_BASELINE_YEARS=3           # Years of historical data for baselines
CONFIDENCE_THRESHOLD=0.7            # Minimum confidence for change detection
```

### **📊 Configuration Options**

| Setting | Description | Default | Options |
|---------|-------------|---------|---------|
| `CHANGE_THRESHOLD` | Sensitivity for change detection | `0.3` | `0.1-0.9` |
| `CONFIDENCE_THRESHOLD` | Minimum confidence for alerts | `0.7` | `0.5-0.95` |
| `MAX_AOI_SIZE_KM2` | Maximum area size (km²) | `1000` | `1-10000` |
| `TEMPORAL_BASELINE_YEARS` | Historical baseline period | `3` | `1-10` |
| `ALERT_EMAIL_COOLDOWN_HOURS` | Alert frequency limit | `24` | `1-168` |

---

## 🧪 Testing

### **🚀 Run All Tests**

```bash
# Comprehensive test suite
python -m pytest tests/ -v

# Quick functionality check
python test_basic_functionality.py

# System verification
python verify_installation.py
```

### **🔬 Test Categories**

```bash
# Unit tests - individual components
python -m pytest tests/unit/ -v

# Integration tests - full workflows  
python -m pytest tests/integration/ -v

# Performance tests
python -m pytest tests/performance/ -v

# Test specific functionality
python -m pytest tests/unit/test_spectral_indices.py -v
python -m pytest tests/integration/test_full_pipeline.py -v
```

### **📊 Test Coverage**

```bash
# Generate coverage report
pip install coverage
coverage run -m pytest
coverage report -m
coverage html  # Creates htmlcov/index.html
```

Expected test results:
- ✅ **95%+ test coverage**
- ✅ **All critical paths tested**  
- ✅ **ML models validated**
- ✅ **API endpoints verified**
- ✅ **Database operations tested**

---

## 🚀 Deployment

### **🐳 Docker Deployment (Recommended)**

```bash
# Production deployment with Docker
cp config/.env.example .env
# Configure .env for production

# Deploy with Docker Compose
docker-compose -f config/docker-compose.yml up -d

# Initialize database
docker-compose exec web python manage.py migrate
docker-compose exec web python manage.py createsuperuser
docker-compose exec web python manage.py collectstatic --noinput
```

### **☁️ Cloud Deployment**

#### **AWS Deployment**
```bash
# Using AWS ECS/Fargate
aws ecs create-cluster --cluster-name change-detection
# Configure RDS PostgreSQL with PostGIS
# Set up S3 for media storage
# Configure CloudWatch for monitoring
```

#### **Google Cloud Platform**
```bash
# Using Cloud Run
gcloud run deploy change-detection \
  --image gcr.io/YOUR_PROJECT/change-detection \
  --platform managed \
  --region us-central1
```

#### **Azure Deployment**
```bash
# Using Azure Container Instances
az container create \
  --resource-group myResourceGroup \
  --name change-detection \
  --image myregistry.azurecr.io/change-detection:latest
```

### **🔧 Production Checklist**

- [ ] Configure PostgreSQL with PostGIS
- [ ] Set up Redis for Celery tasks
- [ ] Configure email SMTP settings  
- [ ] Set up SSL certificates
- [ ] Configure reverse proxy (nginx)
- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Configure log aggregation
- [ ] Set up backup procedures
- [ ] Configure API rate limiting
- [ ] Set up health checks

---

## 💻 CLI Usage

The system includes a powerful command-line interface for automation and scripting.

### **📊 AOI Management**

```bash
# List all AOIs
python src/manage.py shell -c "
from changedetection.cli import AOICLIManager
manager = AOICLIManager()
manager.list_aois()
"

# Create AOI from GeoJSON
python src/changedetection/cli.py create-aoi \
  --name "Forest Monitor" \
  --geojson "path/to/forest.geojson" \
  --user admin

# Monitor AOI for changes
python src/changedetection/cli.py monitor-aoi \
  --aoi-id 1 \
  --threshold 0.3 \
  --email-alerts
```

### **🛰️ Satellite Data Processing**

```bash
# Download satellite data for AOI
python src/changedetection/cli.py download-data \
  --aoi-id 1 \
  --start-date 2023-01-01 \
  --end-date 2023-12-31 \
  --satellites sentinel2,landsat8

# Run change detection
python src/changedetection/cli.py detect-changes \
  --aoi-id 1 \
  --before-date 2023-01-01 \
  --after-date 2023-06-01 \
  --output-dir ./results/
```

### **📈 Analysis & Reporting**

```bash
# Generate change report
python src/changedetection/cli.py generate-report \
  --aoi-id 1 \
  --format pdf \
  --include-maps \
  --output report.pdf

# Export results to GIS formats
python src/changedetection/cli.py export-results \
  --aoi-id 1 \
  --format shapefile \
  --output changes.shp
```

### **🔄 Batch Operations**

```bash
# Process multiple AOIs
python src/changedetection/cli.py batch-process \
  --config batch_config.json \
  --parallel 4

# Automated monitoring script
#!/bin/bash
# daily_monitoring.sh
python src/changedetection/cli.py monitor-all-aois \
  --threshold 0.3 \
  --send-alerts \
  --log-level INFO
```

---

## 🌐 Web Interface

### **🏠 Dashboard Features**

Access the web interface at `http://localhost:8000`

**Main Dashboard:**
- 📊 Real-time AOI status overview
- 📈 Change detection statistics  
- 🗺️ Interactive map with AOI visualization
- ⚠️ Recent alerts and notifications
- 📋 Processing job status

**AOI Management:**
- ✏️ Create AOIs by drawing on map
- 📁 Import AOIs from GeoJSON/Shapefile
- ⚙️ Configure monitoring parameters
- 📊 View historical change data
- 📧 Set up email alerts

**Change Detection:**
- 🔍 Visual before/after image comparison
- 📊 Confidence scoring and explainability
- 📈 Temporal analysis charts
- 🎯 Customizable detection thresholds
- 📊 Performance metrics display

**Data Export:**
- 📊 Export change data to multiple formats
- 🗺️ Generate publication-ready maps
- 📋 Create detailed PDF reports
- 📈 Download analysis charts

### **🔐 Authentication & Authorization**

```python
# User roles and permissions
ADMIN_USER     # Full system access
ANALYST_USER   # Create/modify AOIs, run analysis
VIEWER_USER    # Read-only access to results

# API Authentication
Authorization: Token your-api-token-here
```

### **📱 Mobile-Responsive Design**

The web interface is fully responsive and works on:
- 🖥️ Desktop computers
- 💻 Laptops and tablets  
- 📱 Mobile phones
- 📺 Large displays

---

## 🔄 End-to-End Workflow

### **Complete Change Detection Pipeline**

```mermaid
sequenceDiagram
    participant User
    participant WebUI
    participant API
    participant ML
    participant Satellite
    participant Alerts
    
    User->>WebUI: 1. Create AOI
    WebUI->>API: Store AOI geometry
    
    User->>WebUI: 2. Configure monitoring
    WebUI->>API: Set thresholds & alerts
    
    API->>Satellite: 3. Download imagery
    Satellite-->>API: Return satellite data
    
    API->>ML: 4. Run change detection
    ML->>ML: Process spectral indices
    ML->>ML: Apply temporal analysis
    ML->>ML: Generate confidence scores
    ML-->>API: Return change results
    
    API->>Alerts: 5. Check alert thresholds
    Alerts->>User: Send email notification
    
    User->>WebUI: 6. View results
    WebUI->>API: Retrieve change data
    API-->>WebUI: Return maps & statistics
    
    User->>WebUI: 7. Export results
    WebUI->>API: Generate GIS exports
    API-->>User: Download files
```

### **📋 Step-by-Step Tutorial**

#### **1. Create Your First AOI**

```python
# Web Interface
1. Navigate to http://localhost:8000
2. Click "Create AOI" 
3. Draw polygon on map
4. Set name and description
5. Save AOI

# CLI Alternative  
python src/changedetection/cli.py create-aoi \
  --name "My Forest" \
  --lat 40.7128 --lng -74.0060 \
  --radius 1000  # meters
```

#### **2. Configure Change Detection**

```python
# Set detection parameters
aoi_config = {
    "change_threshold": 0.3,        # Sensitivity (0.1-0.9)
    "confidence_threshold": 0.7,    # Minimum confidence
    "alert_email": "user@example.com",
    "temporal_baseline_years": 3,   # Historical baseline
    "spectral_indices": ["ndvi", "ndbi", "ndwi"]
}
```

#### **3. Run Change Detection**

```bash
# Automatic monitoring (recommended)
python src/manage.py run_scheduler  # Runs in background

# Manual detection
python src/changedetection/cli.py detect-changes \
  --aoi-id 1 \
  --before-date 2023-01-01 \
  --after-date 2023-06-01
```

#### **4. Review Results**

```python
# Web interface shows:
- Before/after image pairs
- Change probability maps  
- Confidence scores
- Affected area statistics
- Temporal trend analysis

# Export options:
- Shapefile (.shp)
- GeoTIFF (.tif) 
- KML/KMZ (.kml)
- PDF reports (.pdf)
```

#### **5. Set Up Alerts**

```python
# Email alert configuration
alert_config = {
    "threshold": 0.3,           # Alert when >30% change detected
    "cooldown_hours": 24,       # Max 1 alert per day
    "include_maps": True,       # Attach visual maps
    "include_statistics": True   # Include change statistics
}
```

### **🔄 Automated Workflows**

```bash
# Set up automated monitoring
# crontab -e
0 6 * * * /path/to/daily_monitoring.sh    # Daily at 6 AM
0 0 * * 1 /path/to/weekly_report.sh       # Weekly on Monday
```

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### **🚀 Development Setup**

```bash
# Fork and clone
git clone https://github.com/your-username/change-detection-system.git
cd change-detection-system

# Set up development environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install

# Run tests
python -m pytest tests/
```

### **📝 Contribution Guidelines**

1. **🍴 Fork** the repository
2. **🌿 Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **✅ Test** your changes: `python -m pytest`
4. **📝 Document** your changes
5. **✨ Commit** with clear messages: `git commit -m 'Add amazing feature'`
6. **🚀 Push** to your branch: `git push origin feature/amazing-feature`
7. **🔄 Create** a Pull Request

### **🎯 Areas for Contribution**

- 🧠 **Machine Learning**: New models and algorithms
- 🛰️ **Satellite APIs**: Additional data source integrations
- 🎨 **UI/UX**: Frontend improvements and new features
- 📊 **Analysis**: New spectral indices and temporal methods
- 🧪 **Testing**: Expand test coverage and scenarios
- 📚 **Documentation**: Tutorials, guides, and examples
- 🐳 **DevOps**: Deployment and infrastructure improvements

### **🔄 Development Workflow**

```bash
# Feature development cycle
git checkout main
git pull origin main
git checkout -b feature/new-feature

# Make changes
# Write tests
python -m pytest tests/

# Check code style
black src/
flake8 src/
mypy src/

# Commit and push
git add .
git commit -m "descriptive commit message"
git push origin feature/new-feature
```

### **📊 Code Quality Standards**

- **Test Coverage**: Maintain >90% coverage
- **Code Style**: Follow PEP 8 with Black formatting
- **Type Hints**: Use type annotations where applicable
- **Documentation**: Document all public APIs
- **Performance**: Profile performance-critical code

---

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Change Detection System

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 🙏 Acknowledgments

- **🛰️ Satellite Data**: Sentinel-2 (ESA), Landsat-8 (NASA/USGS), Bhoonidhi (ISRO)
- **🧠 ML Frameworks**: PyTorch, scikit-learn, TensorFlow
- **🌍 Geospatial**: GDAL, Rasterio, GeoPandas, PostGIS
- **🎨 Visualization**: Leaflet, Plotly, Matplotlib
- **🏗️ Framework**: Django, Django REST Framework

---

## 📞 Support

- **📧 Email**: support@changedetection.com
- **💬 Issues**: [GitHub Issues](https://github.com/your-username/change-detection-system/issues)
- **📚 Docs**: [Documentation Wiki](https://github.com/your-username/change-detection-system/wiki)
- **💭 Discussions**: [GitHub Discussions](https://github.com/your-username/change-detection-system/discussions)

---

<div align="center">

**⭐ Star this repository if you find it useful!**

Made with ❤️ for environmental monitoring and change detection

</div> 