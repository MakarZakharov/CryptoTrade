#!/usr/bin/env python3
"""
Environment Setup Script

Sets up the development and production environment for the trading DRL project.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(command: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command"""
    print(f"Running: {command}")
    return subprocess.run(command, shell=True, check=check)


def setup_python_environment(env_type: str = "dev") -> None:
    """Setup Python virtual environment and install dependencies"""
    print(f"Setting up Python environment for {env_type}...")
    
    # Create virtual environment if it doesn't exist
    if not Path(".venv").exists():
        run_command(f"{sys.executable} -m venv .venv")
    
    # Activate virtual environment and install dependencies
    if os.name == 'nt':  # Windows
        pip_path = ".venv/Scripts/pip"
    else:  # Unix/Linux/macOS
        pip_path = ".venv/bin/pip"
    
    run_command(f"{pip_path} install --upgrade pip")
    run_command(f"{pip_path} install -r requirements/{env_type}.txt")


def setup_pre_commit() -> None:
    """Setup pre-commit hooks"""
    print("Setting up pre-commit hooks...")
    run_command("pre-commit install")


def setup_docker() -> None:
    """Setup Docker environment"""
    print("Setting up Docker environment...")
    run_command("docker-compose build")


def setup_database() -> None:
    """Setup database"""
    print("Setting up database...")
    run_command("docker-compose up -d postgres")
    # Wait for database to be ready
    import time
    time.sleep(10)
    run_command("python scripts/init_database.py")


def run_tests() -> None:
    """Run test suite"""
    print("Running tests...")
    run_command("pytest --cov=trading_drl_project tests/")


def main():
    parser = argparse.ArgumentParser(description="Setup trading DRL project environment")
    parser.add_argument(
        "--env", 
        choices=["dev", "prod", "gpu"], 
        default="dev",
        help="Environment type to setup"
    )
    parser.add_argument(
        "--skip-docker", 
        action="store_true",
        help="Skip Docker setup"
    )
    parser.add_argument(
        "--skip-tests", 
        action="store_true",
        help="Skip running tests"
    )
    
    args = parser.parse_args()
    
    print("🚀 Setting up Trading DRL Project Environment")
    print("=" * 50)
    
    try:
        # Setup Python environment
        setup_python_environment(args.env)
        
        # Setup pre-commit hooks
        setup_pre_commit()
        
        # Setup Docker (unless skipped)
        if not args.skip_docker:
            setup_docker()
            setup_database()
        
        # Run tests (unless skipped)
        if not args.skip_tests:
            run_tests()
        
        print("\n✅ Environment setup completed successfully!")
        print("\nNext steps:")
        print("1. Activate virtual environment: source .venv/bin/activate")
        print("2. Start services: docker-compose up -d")
        print("3. Run the application: python -m trading_drl_project")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()