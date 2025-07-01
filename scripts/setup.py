#!/usr/bin/env python
"""
Setup script for the Change Detection System
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, check=True):
    """Run a shell command"""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, check=check)
    return result.returncode == 0

def setup_environment():
    """Set up the development environment"""
    print("Setting up Change Detection System...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    
    # Create directories
    directories = [
        'media',
        'staticfiles', 
        'logs',
        'satellite_data',
        'change_maps',
        'ml_models'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Check for .env file
    if not os.path.exists('.env'):
        if os.path.exists('.env.example'):
            shutil.copy('.env.example', '.env')
            print("Created .env file from .env.example")
            print("Please edit .env file with your configuration")
        else:
            # Create basic .env file
            env_content = """DEBUG=True
SECRET_KEY=django-insecure-change-me-in-production
DATABASE_URL=postgis://postgres:password@localhost:5432/changedetection
REDIS_URL=redis://localhost:6379/0
ALLOWED_HOSTS=localhost,127.0.0.1,0.0.0.0

# Email settings (configure for production)
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_HOST_USER=your-email@gmail.com
EMAIL_HOST_PASSWORD=your-password
EMAIL_USE_TLS=True
DEFAULT_FROM_EMAIL=noreply@changedetection.com

# Satellite API credentials (get from providers)
SENTINEL_HUB_CLIENT_ID=
SENTINEL_HUB_CLIENT_SECRET=
SENTINEL_HUB_INSTANCE_ID=
USGS_USERNAME=
USGS_PASSWORD=
"""
            with open('.env', 'w') as f:
                f.write(env_content)
            print("Created basic .env file - please configure with your settings")
    
    print("\nSetup completed successfully!")
    print("\nNext steps:")
    print("1. Configure your .env file with API credentials")
    print("2. Start the database: docker-compose up db redis -d")
    print("3. Run migrations: python manage.py migrate")
    print("4. Create superuser: python manage.py createsuperuser")
    print("5. Start the development server: python manage.py runserver")

def setup_database():
    """Set up the database"""
    print("Setting up database...")
    
    # Run migrations
    if not run_command("python manage.py migrate"):
        print("Error running migrations")
        return False
    
    # Create superuser if needed
    create_superuser = input("Create superuser account? (y/n): ").lower() == 'y'
    if create_superuser:
        run_command("python manage.py createsuperuser", check=False)
    
    return True

def collect_static():
    """Collect static files"""
    print("Collecting static files...")
    return run_command("python manage.py collectstatic --noinput")

def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup Change Detection System')
    parser.add_argument('--env-only', action='store_true', 
                       help='Only setup environment, skip database')
    parser.add_argument('--db-only', action='store_true',
                       help='Only setup database')
    
    args = parser.parse_args()
    
    if args.db_only:
        setup_database()
    elif args.env_only:
        setup_environment()
    else:
        setup_environment()
        
        # Ask if user wants to setup database
        setup_db = input("Setup database now? (requires running PostgreSQL) (y/n): ").lower() == 'y'
        if setup_db:
            setup_database()
            collect_static()

if __name__ == '__main__':
    main() 