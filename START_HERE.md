# 🚀 Quick Start Guide

Get the Robust Change Detection System running in **5 minutes**!

## ⚡ Instant Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup Database
```bash
python manage.py migrate
python manage.py createsuperuser
# Enter: username=admin, password=admin123
```

### 3. Start the System
```bash
python manage.py runserver
```

### 4. Access the Application
- **Web Interface**: http://localhost:8000
- **Admin Panel**: http://localhost:8000/admin (admin/admin123)
- **Dashboard**: http://localhost:8000/dashboard

## 🎯 Test the System

### Option 1: Web Demo
1. Go to http://localhost:8000
2. Click "Create Area of Interest" 
3. Draw an area on the map
4. Upload before/after images
5. View change detection results

### Option 2: Command Line Demo
```bash
python demo_change_detection.py
```
This runs automated tests with:
- 🌳 Deforestation scenarios
- 🏢 Urban development 
- 🚜 Agricultural expansion
- 📊 Professional analysis reports

## 🔧 Next Steps

### Enable Background Processing
```bash
# Install Redis (Windows)
# Download from: https://github.com/microsoftarchive/redis/releases

# Start Celery worker
celery -A backend worker --loglevel=info
```

### Setup Email Alerts
1. Copy `env_example.txt` to `.env`
2. Add your email credentials
3. Restart the server

### Production Deployment
```bash
docker-compose up -d
```

## 📊 What You'll See

The system will demonstrate:
- ✅ **Multi-temporal analysis** of satellite imagery
- ✅ **Automated change detection** with confidence scores
- ✅ **Interactive maps** for AOI selection
- ✅ **Professional reports** with visualizations
- ✅ **Real-time processing** (< 1 second per image pair)
- ✅ **Email alerts** for significant changes

## 🎉 Success Indicators

You'll know it's working when you see:
- 🌐 Web interface loads at localhost:8000
- 🗺️ Interactive map displays correctly
- 🔍 Change detection processes images
- 📊 Results show percentage changes
- 📧 Email notifications (if configured)

## 🆘 Need Help?

- Check `README.md` for detailed documentation
- Run `python demo_change_detection.py` for validation
- Visit the admin panel to verify data models
- Check console logs for any error messages

**Ready to monitor environmental changes!** 🌍✨ 