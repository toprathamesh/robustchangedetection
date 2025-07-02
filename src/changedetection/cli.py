"""
Command-line interface for the Change Detection System.
Provides convenient commands for common operations.
"""

import argparse
import sys
import os
from pathlib import Path
import django
from django.core.management import execute_from_command_line
from django.conf import settings

# Add the src directory to Python path
src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

# Configure Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'web.backend.settings')
django.setup()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Change Detection System CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available commands:
  startserver     Start the Django development server
  migrate         Run database migrations
  createsuperuser Create a Django superuser
  test            Run the test suite
  worker          Start a Celery worker
  shell           Start Django shell
  collectstatic   Collect static files for production

Examples:
  change-detection startserver
  change-detection migrate
  change-detection test
  change-detection worker
        """
    )
    
    parser.add_argument(
        'command',
        help='Command to execute',
        choices=[
            'startserver', 'migrate', 'createsuperuser', 'test', 
            'worker', 'shell', 'collectstatic', 'version'
        ]
    )
    
    parser.add_argument(
        '--port',
        default='8000',
        help='Port for development server (default: 8000)'
    )
    
    parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Host for development server (default: 127.0.0.1)'
    )
    
    parser.add_argument(
        '--settings',
        help='Django settings module to use'
    )
    
    args = parser.parse_args()
    
    # Override settings module if specified
    if args.settings:
        os.environ['DJANGO_SETTINGS_MODULE'] = args.settings
    
    # Change to src directory for Django commands
    src_path = Path(__file__).parent.parent
    os.chdir(src_path)
    
    try:
        if args.command == 'startserver':
            print(f"🚀 Starting development server on {args.host}:{args.port}")
            print(f"📊 Admin interface: http://{args.host}:{args.port}/admin/")
            print(f"📖 API documentation: http://{args.host}:{args.port}/api/docs/")
            execute_from_command_line([
                'manage.py', 'runserver', f'{args.host}:{args.port}'
            ])
            
        elif args.command == 'migrate':
            print("🔄 Running database migrations...")
            execute_from_command_line(['manage.py', 'migrate'])
            print("✅ Migrations completed successfully")
            
        elif args.command == 'createsuperuser':
            print("👤 Creating superuser...")
            execute_from_command_line(['manage.py', 'createsuperuser'])
            
        elif args.command == 'test':
            print("🧪 Running test suite...")
            # Change back to project root for pytest
            os.chdir(src_path.parent)
            import subprocess
            result = subprocess.run(['pytest', '-v'], capture_output=False)
            sys.exit(result.returncode)
            
        elif args.command == 'worker':
            print("⚙️  Starting Celery worker...")
            import subprocess
            result = subprocess.run([
                'celery', '-A', 'web.backend', 'worker', '-l', 'info'
            ], capture_output=False)
            sys.exit(result.returncode)
            
        elif args.command == 'shell':
            print("🐚 Starting Django shell...")
            execute_from_command_line(['manage.py', 'shell'])
            
        elif args.command == 'collectstatic':
            print("📦 Collecting static files...")
            execute_from_command_line(['manage.py', 'collectstatic', '--noinput'])
            print("✅ Static files collected successfully")
            
        elif args.command == 'version':
            from . import __version__
            print(f"Change Detection System v{__version__}")
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def setup_environment():
    """Set up environment for running the application."""
    print("🔧 Setting up Change Detection System environment...")
    
    # Check for required environment variables
    required_vars = [
        'SECRET_KEY', 'DATABASE_URL', 'REDIS_URL'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("⚠️  Warning: Missing environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n💡 Copy config/.env.example to .env and configure your settings")
    
    # Check database connection
    try:
        from django.db import connection
        connection.ensure_connection()
        print("✅ Database connection successful")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("💡 Make sure PostgreSQL is running and DATABASE_URL is correct")
    
    # Check Redis connection
    try:
        import redis
        redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
        r = redis.from_url(redis_url)
        r.ping()
        print("✅ Redis connection successful")
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print("💡 Make sure Redis is running and REDIS_URL is correct")


if __name__ == '__main__':
    main() 