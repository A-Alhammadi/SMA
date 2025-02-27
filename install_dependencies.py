#!/usr/bin/env python3
# install_dependencies.py - Script to install required dependencies

import subprocess
import sys
import importlib.util

def is_package_installed(package_name):
    """Check if a package is installed"""
    try:
        spec = importlib.util.find_spec(package_name)
        return spec is not None
    except (ModuleNotFoundError, ValueError):
        return False

def install_package(package_name):
    """Install a package using pip"""
    print(f"Installing {package_name}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

def main():
    """Check and install required dependencies"""
    print("Checking and installing required dependencies...")
    
    # List of required packages
    required_packages = [
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "scipy",
        "statsmodels",
        "hmmlearn",  # For Hidden Markov Models
        "arch",      # For GARCH volatility models
        "joblib",    # For model serialization
        "psycopg2-binary",  # For PostgreSQL database access
    ]
    
    # Check and install each package
    for package in required_packages:
        if not is_package_installed(package.split("==")[0] if "==" in package else package):
            try:
                install_package(package)
                print(f"Successfully installed {package}")
            except Exception as e:
                print(f"Failed to install {package}: {e}")
        else:
            print(f"{package} is already installed")
    
    print("\nDependency installation completed!")
    print("You can now run the enhanced_sma strategy.")

if __name__ == "__main__":
    main()