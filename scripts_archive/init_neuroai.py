#!/usr/bin/env python
"""
NeuroAI-Handbook Environment Setup Script
-----------------------------------------

This script helps setup the environment for working with the NeuroAI Handbook.
It checks for required dependencies, creates necessary directories, and validates
the installation.

Usage:
    python init_neuroai.py [--install-deps] [--check-env] [--create-dirs]

Options:
    --install-deps    Install required dependencies
    --check-env       Check the environment for required packages
    --create-dirs     Create standard directory structure for exercises
    --all             Perform all setup tasks (default if no flags provided)
"""

import os
import sys
import subprocess
import platform
import argparse
from pathlib import Path


def print_header(message):
    """Print a formatted header message."""
    print("\n" + "=" * 80)
    print(f" {message}")
    print("=" * 80)


def check_python_version():
    """Check if Python version is compatible."""
    print_header("Checking Python Version")
    
    major, minor, *_ = sys.version_info
    print(f"Python version: {sys.version}")
    
    if major < 3 or (major == 3 and minor < 8):
        print("❌ WARNING: Python 3.8 or higher is recommended for this handbook.")
        print(f"   Current version: {major}.{minor}")
        return False
    else:
        print("✅ Python version is compatible.")
        return True


def check_dependencies():
    """Check if required packages are installed."""
    print_header("Checking Dependencies")
    
    required_packages = [
        # Core scientific packages
        "numpy", "scipy", "matplotlib", "pandas", "seaborn", "jupyter",
        # Neuroscience packages
        "mne", "nibabel", "nilearn",
        # Machine learning packages
        "scikit-learn", "torch", "tensorflow",
        # Jupyter Book (for building the handbook)
        "jupyter-book",
    ]
    
    missing_packages = []
    installed_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            installed_packages.append(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} (missing)")
    
    if missing_packages:
        print("\nMissing packages:")
        print("  " + ", ".join(missing_packages))
        print("\nTo install missing packages, run:")
        print("  pip install " + " ".join(missing_packages))
        print("  or")
        print("  python init_neuroai.py --install-deps")
        return False, missing_packages
    else:
        print("\nAll required packages are installed.")
        return True, []


def install_missing_dependencies(missing_packages):
    """Install missing dependencies using pip."""
    print_header("Installing Missing Dependencies")
    
    if not missing_packages:
        print("No missing dependencies to install.")
        return True
    
    print(f"Installing: {', '.join(missing_packages)}")
    
    try:
        # Use pip to install the missing packages
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
        print("\n✅ Successfully installed all missing dependencies.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error installing dependencies: {e}")
        print("Please try installing them manually using:")
        print("  pip install " + " ".join(missing_packages))
        return False


def create_directory_structure():
    """Create a standard directory structure for exercises and data."""
    print_header("Creating Directory Structure")
    
    base_dir = Path.cwd()
    directories = [
        "data/raw",
        "data/processed",
        "exercises",
        "models",
        "results",
        "notebooks"
    ]
    
    created_dirs = []
    
    for directory in directories:
        dir_path = base_dir / directory
        if not dir_path.exists():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                created_dirs.append(str(dir_path))
                print(f"✅ Created: {dir_path}")
            except Exception as e:
                print(f"❌ Error creating {dir_path}: {e}")
        else:
            print(f"ℹ️ Directory already exists: {dir_path}")
    
    if created_dirs:
        print(f"\nCreated {len(created_dirs)} directories.")
    else:
        print("\nNo new directories needed to be created.")
    
    return len(created_dirs) >= 0


def create_readme_files(base_dir):
    """Create README.md files in each directory to explain its purpose."""
    directories = {
        "data/raw": "This directory contains raw, unprocessed data files used in the exercises.",
        "data/processed": "This directory stores processed data files generated during analysis.",
        "exercises": "Contains exercise files and practice problems from the handbook.",
        "models": "Stores trained ML models and related files.",
        "results": "Contains outputs, figures, and results from experiments.",
        "notebooks": "Jupyter notebooks for interactive learning and experimentation."
    }
    
    for directory, description in directories.items():
        readme_path = Path(base_dir) / directory / "README.md"
        if not readme_path.exists():
            try:
                with open(readme_path, 'w') as f:
                    f.write(f"# {directory.split('/')[-1].title()}\n\n{description}\n")
                print(f"✅ Created README in: {directory}")
            except Exception as e:
                print(f"❌ Error creating README in {directory}: {e}")


def validate_installation():
    """Validate the installation by running a simple test."""
    print_header("Validating Installation")
    
    # Test importing essential packages
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        
        # Create a simple plot to test matplotlib
        plt.figure(figsize=(3, 3))
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        plt.plot(x, y)
        plt.title("Test Plot")
        
        # Save the plot
        test_dir = Path("results")
        if not test_dir.exists():
            test_dir.mkdir(parents=True, exist_ok=True)
        
        plt.savefig("results/test_plot.png")
        plt.close()
        
        print("✅ NumPy and Matplotlib are working correctly.")
        print("✅ Successfully saved a test plot to results/test_plot.png")
        return True
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Setup environment for NeuroAI Handbook")
    parser.add_argument('--install-deps', action='store_true', help='Install required dependencies')
    parser.add_argument('--check-env', action='store_true', help='Check the environment')
    parser.add_argument('--create-dirs', action='store_true', help='Create standard directory structure')
    parser.add_argument('--all', action='store_true', help='Perform all setup tasks')
    
    args = parser.parse_args()
    
    # If no arguments are provided, perform all tasks
    if not (args.install_deps or args.check_env or args.create_dirs):
        args.all = True
    
    print_header("NeuroAI Handbook Setup")
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Working directory: {os.getcwd()}")
    
    # Check Python version in all cases
    python_ok = check_python_version()
    
    # Check dependencies
    if args.check_env or args.all or args.install_deps:
        deps_ok, missing_packages = check_dependencies()
        
        # Install missing dependencies if requested
        if args.install_deps or args.all:
            if missing_packages:
                install_missing_dependencies(missing_packages)
    
    # Create directory structure if requested
    if args.create_dirs or args.all:
        dirs_created = create_directory_structure()
        if dirs_created:
            create_readme_files(os.getcwd())
    
    # Validate installation if all tasks are performed
    if args.all:
        validate_installation()
    
    print_header("Setup Complete")
    print("For more information, check out the README.md file or visit the handbook online.")
    print("To build the book locally, run: jb build book")
    print("To view the book in a browser: python -m http.server -d book/_build/html")


if __name__ == "__main__":
    main()