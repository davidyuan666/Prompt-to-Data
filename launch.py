#!/usr/bin/env python3
"""
Prompt-to-Data Framework Launcher
A user-friendly interface to run different components of the P2D framework.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def print_banner():
    """Print the application banner."""
    banner = """
╔══════════════════════════════════════════╗
║      Prompt-to-Data Framework v1.0       ║
║      Data Synthesis & Analysis Tool      ║
╚══════════════════════════════════════════╝
    """
    print(banner)

def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    required_packages = ['openai', 'tqdm', 'pandas']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("✅ All dependencies are installed.")
        return True

def setup_environment():
    """Setup environment variables for API keys."""
    print("\n🔧 Environment Setup")
    print("-" * 40)
    
    # Check for existing API key
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if api_key:
        print(f"✅ DEEPSEEK_API_KEY found in environment")
        return True
    
    print("⚠️  No DEEPSEEK_API_KEY found in environment variables.")
    print("\nTo set up your API key:")
    print("1. Create a .env file with: DEEPSEEK_API_KEY=your_key_here")
    print("2. Or set it manually: set DEEPSEEK_API_KEY=your_key_here (Windows)")
    print("3. Or edit the API_KEY variable in the Python scripts directly")
    
    create_env = input("\nCreate .env file? (y/n): ").lower()
    if create_env == 'y':
        api_key = input("Enter your DeepSeek API key: ").strip()
        with open('.env', 'w') as f:
            f.write(f"DEEPSEEK_API_KEY={api_key}\n")
        print("✅ .env file created. Remember to add it to .gitignore!")
        return True
    
    return False

def run_synthesis():
    """Run the main synthesis pipeline."""
    print("\n🚀 Starting P2D Synthesis Pipeline")
    print("-" * 40)
    
    # Check if ds1000.jsonl exists
    if not Path("ds1000.jsonl").exists():
        print("⚠️  Warning: ds1000.jsonl not found in current directory.")
        print("The script will use mock data for demonstration.")
    
    print("\nRunning p2d_synthesis.py...")
    print("This will generate training data in ChatML format.")
    print("Output file: qwen_finetune_data.jsonl")
    
    confirm = input("\nContinue? (y/n): ").lower()
    if confirm == 'y':
        subprocess.run([sys.executable, "p2d_synthesis.py"])
    else:
        print("Operation cancelled.")

def run_experiment():
    """Run the experiment and report generation."""
    print("\n📊 Starting P2D Experiment Pipeline")
    print("-" * 40)
    
    if not Path("ds1000.jsonl").exists():
        print("❌ Error: ds1000.jsonl is required but not found.")
        print("Please ensure the dataset file is in the current directory.")
        return
    
    print("\nRunning p2d_v1.py...")
    print("This will:")
    print("1. Generate SFT training data (p2d_sft_train.jsonl)")
    print("2. Create experiment report (p2d_experiment_report.json/csv)")
    print("3. Analyze token metrics and format compliance")
    
    confirm = input("\nContinue? (y/n): ").lower()
    if confirm == 'y':
        subprocess.run([sys.executable, "p2d_v1.py"])
    else:
        print("Operation cancelled.")

def install_dependencies():
    """Install project dependencies."""
    print("\n📦 Installing Dependencies")
    print("-" * 40)
    
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found!")
        return
    
    print("Installing packages from requirements.txt...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")

def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(description="Prompt-to-Data Framework Launcher")
    parser.add_argument('--setup', action='store_true', help='Setup environment only')
    parser.add_argument('--synthesis', action='store_true', help='Run synthesis pipeline directly')
    parser.add_argument('--experiment', action='store_true', help='Run experiment pipeline directly')
    parser.add_argument('--install', action='store_true', help='Install dependencies only')
    
    args = parser.parse_args()
    
    # Handle command line arguments
    if args.setup:
        setup_environment()
        return
    elif args.synthesis:
        run_synthesis()
        return
    elif args.experiment:
        run_experiment()
        return
    elif args.install:
        install_dependencies()
        return
    
    # Interactive mode
    print_banner()
    
    while True:
        print("\n" + "=" * 50)
        print("MAIN MENU")
        print("=" * 50)
        print("1. 🔧 Setup Environment & Check Dependencies")
        print("2. 🚀 Run Synthesis Pipeline (p2d_synthesis.py)")
        print("3. 📊 Run Experiment Pipeline (p2d_v1.py)")
        print("4. 📦 Install/Update Dependencies")
        print("5. 📖 View Documentation")
        print("6. 🚪 Exit")
        print("=" * 50)
        
        try:
            choice = input("\nSelect an option (1-6): ").strip()
            
            if choice == '1':
                setup_environment()
                check_dependencies()
            elif choice == '2':
                if check_dependencies():
                    run_synthesis()
            elif choice == '3':
                if check_dependencies():
                    run_experiment()
            elif choice == '4':
                install_dependencies()
            elif choice == '5':
                print("\n📖 Documentation:")
                print("-" * 40)
                print("Project: Prompt-to-Data Framework")
                print("Main Files:")
                print("  • p2d_synthesis.py - Main synthesis pipeline")
                print("  • p2d_v1.py - Experiment and report generation")
                print("  • ds1000.jsonl - Input dataset (DS-1000)")
                print("\nUsage:")
                print("  python launch.py --help  # Show all options")
                print("  python launch.py --setup # Setup environment")
                print("\nAPI Configuration:")
                print("  Set DEEPSEEK_API_KEY in .env file or environment")
            elif choice == '6':
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please select 1-6.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()