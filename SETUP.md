# Robust Change Detection System - Setup Guide

This guide will help you set up and deploy the satellite-based change detection system.

## Prerequisites

### Required Software
- **Python 3.8+**: Main programming language
- **Docker & Docker Compose**: For containerized deployment
- **PostgreSQL with PostGIS**: Spatial database (can use Docker)
- **Redis**: For task queue (can use Docker)

### Optional (for development)
- **Git**: Version control
- **Node.js**: For any frontend build tools
- **GDAL**: Geospatial libraries (included in Docker)

## Quick Start with Docker (Recommended)

### 1. Clone and Setup
```bash
git clone <repository-url>
cd robustchangedetection

# Create environment file
cp .env.example .env
# Edit .env with your configuration
```

### 2. Configure Environment Variables
Edit the `.env` file with your settings:

```bash
# Django settings
DEBUG=True
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=localhost,127.0.0.1,0.0.0.0

# Database (automatically configured for Docker)
DATABASE_URL=postgis://postgres:password@db:5432/changedetection

# Satellite API credentials (REQUIRED for functionality)
SENTINEL_HUB_CLIENT_ID=your-sentinel-hub-client-id
SENTINEL_HUB_CLIENT_SECRET=your-sentinel-hub-client-secret
USGS_USERNAME=your-usgs-username
USGS_PASSWORD=your-usgs-password

# Email settings (REQUIRED for alerts)
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_HOST_USER=your-email@gmail.com
EMAIL_HOST_PASSWORD=your-app-password
DEFAULT_FROM_EMAIL=your-email@gmail.com
```

### 3. Start the Application
```bash
# Build and start all services
docker-compose up --build

# The application will be available at http://localhost:8000
```

### 4. Initialize Database
```bash
# Run migrations
docker-compose exec web python manage.py migrate

# Create superuser
docker-compose exec web python manage.py createsuperuser

# Load sample data (optional)
docker-compose exec web python manage.py loaddata sample_data.json
```

## Manual Setup (Development)

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run setup script
python scripts/setup.py
```

### 2. Database Setup
```bash
# Install PostgreSQL with PostGIS
# Ubuntu/Debian:
sudo apt-get install postgresql postgresql-contrib postgis

# Create database
sudo -u postgres psql
CREATE DATABASE changedetection;
CREATE USER changeuser WITH PASSWORD 'password';
GRANT ALL PRIVILEGES ON DATABASE changedetection TO changeuser;
ALTER DATABASE changedetection OWNER TO changeuser;

# Enable PostGIS
\c changedetection
CREATE EXTENSION postgis;
\q
```

### 3. Redis Setup
```bash
# Ubuntu/Debian:
sudo apt-get install redis-server
sudo systemctl start redis-server

# Or use Docker:
docker run -d -p 6379:6379 redis:alpine
```

### 4. Run Development Server
```bash
# Migrate database
python manage.py migrate

# Create superuser
python manage.py createsuperuser

# Collect static files
python manage.py collectstatic

# Start development server
python manage.py runserver

# In separate terminals, start Celery:
celery -A backend worker --loglevel=info
celery -A backend beat --loglevel=info
```

## API Credentials Setup

### Sentinel Hub API
1. Register at [Sentinel Hub](https://www.sentinel-hub.com/)
2. Create a new configuration
3. Get your Client ID, Client Secret, and Instance ID
4. Add to `.env` file

### USGS Earth Explorer API
1. Register at [USGS Earth Explorer](https://earthexplorer.usgs.gov/)
2. Apply for machine-to-machine access
3. Add username and password to `.env` file

## Configuration Options

### Application Settings
```python
# In backend/settings.py or via environment variables

# Maximum AOI size in square kilometers
MAX_AOI_SIZE_KM2 = 1000

# Maximum processing time for jobs
MAX_PROCESSING_TIME_MINUTES = 60

# Default change detection threshold
CHANGE_THRESHOLD = 0.3

# Alert cooldown period
ALERT_EMAIL_COOLDOWN_HOURS = 24
```

### Model Configuration
```python
# Available models
MODEL_OPTIONS = {
    'unet_change_detection': {
        'name': 'U-Net Change Detection',
        'input_channels': 6,  # Before + After RGB
        'output_classes': 1,  # Binary change map
    }
}
```

## Usage Guide

### 1. Access the Application
- Web Interface: http://localhost:8000
- Admin Interface: http://localhost:8000/admin
- API Documentation: http://localhost:8000/api/docs

### 2. Create Areas of Interest (AOI)
1. Login to the web interface
2. Navigate to the Interactive Map
3. Use drawing tools to create polygons
4. Configure satellite source and parameters
5. Save the AOI

### 3. Monitor Changes
The system will automatically:
- Check for new satellite imagery
- Download and preprocess images
- Run change detection algorithms
- Send email alerts for significant changes

### 4. View Results
- Dashboard: Overview of all AOIs and recent activity
- AOI Detail Pages: Detailed results for specific areas
- Interactive Map: Visualize change overlays

## API Usage

### Authentication
```python
import requests

# Get token
response = requests.post('http://localhost:8000/api/auth/login/', {
    'username': 'your_username',
    'password': 'your_password'
})
token = response.json()['token']

# Use token in headers
headers = {'Authorization': f'Token {token}'}
```

### Create AOI via API
```python
aoi_data = {
    'name': 'Test AOI',
    'description': 'Test area for monitoring',
    'geometry': {
        'type': 'Polygon',
        'coordinates': [[
            [-120.0, 35.0],
            [-120.1, 35.0], 
            [-120.1, 35.1],
            [-120.0, 35.1],
            [-120.0, 35.0]
        ]]
    },
    'satellite_source': 'sentinel2',
    'cloud_cover_threshold': 20.0
}

response = requests.post(
    'http://localhost:8000/api/core/aois/',
    json=aoi_data,
    headers=headers
)
```

## Model Training

### Prepare Training Data
```bash
# Organize data in this structure:
training_data/
├── train_pairs.txt     # List of image pairs
├── val_pairs.txt       # Validation pairs
├── before/             # Before images
├── after/              # After images
└── masks/              # Change masks
```

### Train Model
```bash
python scripts/train_model.py \
    --data_dir /path/to/training_data \
    --epochs 100 \
    --batch_size 8 \
    --model_dir ml_models
```

## Deployment

### Production Settings
```bash
# Update .env for production
DEBUG=False
SECRET_KEY=your-production-secret-key
ALLOWED_HOSTS=your-domain.com
USE_HTTPS=True

# Use production database
DATABASE_URL=postgis://user:pass@prod-db:5432/changedetection

# Configure email backend
EMAIL_BACKEND=django.core.mail.backends.smtp.EmailBackend

# Set up monitoring
SENTRY_DSN=your-sentry-dsn
```

### Docker Production Deployment
```bash
# Build production image
docker build -t change-detection:prod .

# Run with production compose file
docker-compose -f docker-compose.prod.yml up -d
```

### Scaling
```bash
# Scale web workers
docker-compose up --scale web=3

# Scale Celery workers
docker-compose up --scale celery=5
```

## Monitoring and Maintenance

### Health Checks
- Database: `python manage.py dbshell`
- Redis: `redis-cli ping`
- Celery: Check admin interface at `/admin/django_celery_results/taskresult/`

### Log Monitoring
```bash
# Application logs
tail -f logs/django.log

# Docker logs
docker-compose logs -f web
docker-compose logs -f celery
```

### Database Maintenance
```bash
# Regular backup
pg_dump changedetection > backup_$(date +%Y%m%d).sql

# Cleanup old results
python manage.py shell -c "
from change_detection.models import ChangeDetectionJob
from datetime import datetime, timedelta
old_jobs = ChangeDetectionJob.objects.filter(
    created_at__lt=datetime.now() - timedelta(days=90)
)
old_jobs.delete()
"
```

## Troubleshooting

### Common Issues

1. **GDAL Installation Error**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install gdal-bin libgdal-dev
   
   # macOS
   brew install gdal
   ```

2. **PostGIS Connection Error**
   ```bash
   # Check PostGIS extension
   sudo -u postgres psql changedetection -c "SELECT PostGIS_Version();"
   ```

3. **Celery Not Processing Tasks**
   ```bash
   # Check Redis connection
   redis-cli ping
   
   # Restart Celery workers
   docker-compose restart celery
   ```

4. **Memory Issues with Large Images**
   ```python
   # Reduce batch size in training
   # Implement image tiling for large AOIs
   ```

### Performance Optimization

1. **Database Indexing**
   ```sql
   CREATE INDEX idx_aoi_geometry ON core_areaofinterest USING GIST (geometry);
   ```

2. **Image Caching**
   ```python
   # Use Redis for image caching
   CACHES = {
       'default': {
           'BACKEND': 'django_redis.cache.RedisCache',
           'LOCATION': 'redis://127.0.0.1:6379/1',
       }
   }
   ```

## Support and Contributing

### Getting Help
- Check the logs first: `logs/django.log`
- Review the API documentation: `/api/docs`
- Check GitHub issues

### Contributing
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

### License
This project is licensed under the MIT License. See LICENSE file for details.

## References
- [Sentinel Hub API Documentation](https://docs.sentinel-hub.com/)
- [USGS Earth Explorer](https://earthexplorer.usgs.gov/)
- [Django Documentation](https://docs.djangoproject.com/)
- [PostGIS Documentation](https://postgis.net/documentation/)
- [Satellite Image Deep Learning Techniques](https://github.com/satellite-image-deep-learning/techniques) 