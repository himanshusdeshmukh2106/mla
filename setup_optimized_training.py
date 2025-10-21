"""
Setup Script for Optimized Training
Checks dependencies and prepares environment
"""

import sys
import subprocess
import os


def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"   ❌ Python {version.major}.{version.minor} detected")
        print(f"   ⚠️  Python 3.8+ recommended")
        return False
    print(f"   ✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False


def install_package(package_name):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "-q"])
        return True
    except subprocess.CalledProcessError:
        return False


def check_dependencies():
    """Check and install required dependencies"""
    print("\n📦 Checking dependencies...")
    
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'scikit-learn': 'sklearn',
        'xgboost': 'xgboost',
        'lightgbm': 'lightgbm',
        'catboost': 'catboost',
        'optuna': 'optuna',
        'pandas-ta': 'pandas_ta',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn'
    }
    
    missing_packages = []
    installed_packages = []
    
    for package, import_name in required_packages.items():
        if check_package(package, import_name):
            print(f"   ✅ {package}")
            installed_packages.append(package)
        else:
            print(f"   ❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing {len(missing_packages)} packages")
        print("\nInstall missing packages? (y/n): ", end='')
        response = input().strip().lower()
        
        if response == 'y':
            print("\n📥 Installing missing packages...")
            for package in missing_packages:
                print(f"   Installing {package}...", end=' ')
                if install_package(package):
                    print("✅")
                else:
                    print("❌")
        else:
            print("\n⚠️  Please install manually:")
            print(f"   pip install {' '.join(missing_packages)}")
            return False
    
    return True


def check_data_files():
    """Check if required data files exist"""
    print("\n📁 Checking data files...")
    
    required_files = {
        'ema_crossover_with_targets.csv': 'Training data with targets',
        'create_ema_crossover_targets.py': 'Target generation script'
    }
    
    missing_files = []
    
    for filename, description in required_files.items():
        if os.path.exists(filename):
            print(f"   ✅ {filename}")
        else:
            print(f"   ❌ {filename} ({description})")
            missing_files.append(filename)
    
    if 'ema_crossover_with_targets.csv' in missing_files:
        print("\n⚠️  Training data not found!")
        print("   Run this first: python create_ema_crossover_targets.py")
        return False
    
    return True


def check_disk_space():
    """Check available disk space"""
    print("\n💾 Checking disk space...")
    
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free // (2**30)
        
        if free_gb < 1:
            print(f"   ⚠️  Only {free_gb} GB free")
            print(f"   Recommended: At least 1 GB free")
            return False
        else:
            print(f"   ✅ {free_gb} GB free")
            return True
    except Exception as e:
        print(f"   ⚠️  Could not check disk space: {e}")
        return True


def create_directories():
    """Create necessary directories"""
    print("\n📂 Creating directories...")
    
    directories = ['models', 'results', 'logs']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"   ✅ Created {directory}/")
        else:
            print(f"   ✅ {directory}/ exists")
    
    return True


def show_next_steps():
    """Show next steps to user"""
    print("\n" + "="*80)
    print("🎯 NEXT STEPS")
    print("="*80)
    
    print("\n1️⃣ Analyze your features (optional but recommended):")
    print("   python analyze_ema_crossover_features.py")
    print("   • Shows feature importance")
    print("   • Identifies redundant features")
    print("   • Takes ~2 minutes")
    
    print("\n2️⃣ Train optimized models:")
    print("   python train_ema_crossover_optimized.py")
    print("   • Selects best ~30 features")
    print("   • Trains XGBoost + LightGBM + Ensemble")
    print("   • Runs walk-forward analysis")
    print("   • Takes ~30 minutes")
    
    print("\n3️⃣ Compare with original method:")
    print("   python compare_training_methods.py")
    print("   • Shows improvements")
    print("   • Explains differences")
    
    print("\n4️⃣ Read the guide:")
    print("   OPTIMIZED_TRAINING_GUIDE.md")
    print("   • Detailed documentation")
    print("   • Customization options")
    print("   • Troubleshooting")
    
    print("\n" + "="*80)


def main():
    """Main setup function"""
    print("="*80)
    print("🚀 OPTIMIZED TRAINING SETUP")
    print("="*80)
    
    all_checks_passed = True
    
    # Check Python version
    if not check_python_version():
        all_checks_passed = False
    
    # Check dependencies
    if not check_dependencies():
        all_checks_passed = False
    
    # Check data files
    if not check_data_files():
        all_checks_passed = False
    
    # Check disk space
    if not check_disk_space():
        all_checks_passed = False
    
    # Create directories
    if not create_directories():
        all_checks_passed = False
    
    # Summary
    print("\n" + "="*80)
    if all_checks_passed:
        print("✅ SETUP COMPLETE!")
        print("="*80)
        print("\nYour environment is ready for optimized training!")
        show_next_steps()
    else:
        print("⚠️  SETUP INCOMPLETE")
        print("="*80)
        print("\nPlease resolve the issues above before proceeding.")
        print("\nCommon solutions:")
        print("  • Install missing packages: pip install -r requirements.txt")
        print("  • Generate training data: python create_ema_crossover_targets.py")
        print("  • Free up disk space if needed")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
