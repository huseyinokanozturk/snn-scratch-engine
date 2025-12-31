"""
File Manager Utility

Handles project directory structure and file organization for RHEO SNN.
Ensures all required directories exist and generates .gitignore.

Directories:
- src/environment/maps/: JSON map layouts
- brain_weights/: Saved neural network weights
- experiments/: Simulation stats and performance logs

Usage:
    from src.utils.file_manager import setup_project_structure
    setup_project_structure()
"""
import os
from pathlib import Path
from datetime import datetime


# Project root (relative to this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


# Required directories
REQUIRED_DIRS = [
    "src/environment/maps",
    "brain_weights",
    "experiments",
]


def setup_project_structure():
    """
    Ensure all required directories exist and generate .gitignore.
    Call this at application startup.
    """
    for dir_path in REQUIRED_DIRS:
        full_path = PROJECT_ROOT / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
    
    # Generate .gitignore
    _generate_gitignore()
    
    print(f"Project structure verified: {PROJECT_ROOT}")
    return True


def _generate_gitignore():
    """Generate or update .gitignore file."""
    gitignore_path = PROJECT_ROOT / ".gitignore"
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
dist/
build/

# Virtual environments
venv/
env/
.venv/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Project-specific
brain_weights/*.npz
experiments/*.json
experiments/*.csv

# Keep placeholder files
!brain_weights/.gitkeep
!experiments/.gitkeep
!src/environment/maps/.gitkeep

# OS files
.DS_Store
Thumbs.db
"""
    
    with open(gitignore_path, 'w') as f:
        f.write(gitignore_content)
    
    # Create placeholder files
    for dir_name in ["brain_weights", "experiments", "src/environment/maps"]:
        gitkeep_path = PROJECT_ROOT / dir_name / ".gitkeep"
        gitkeep_path.touch(exist_ok=True)


def get_experiments_path():
    """Get the path to experiments directory."""
    return PROJECT_ROOT / "experiments"


def get_brain_weights_path():
    """Get the path to brain_weights directory."""
    return PROJECT_ROOT / "brain_weights"


def get_maps_path():
    """Get the path to maps directory."""
    return PROJECT_ROOT / "src" / "environment" / "maps"


def get_timestamped_filename(base_name: str, extension: str = ".json") -> str:
    """
    Generate a timestamped filename.
    
    Args:
        base_name: Base name for the file (e.g., "simulation_stats")
        extension: File extension (default: ".json")
    
    Returns:
        Filename with timestamp: "simulation_stats_20231231_223045.json"
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}{extension}"


def list_saved_brains():
    """List all saved brain weight files."""
    brain_path = get_brain_weights_path()
    return sorted(brain_path.glob("*.npz"))


def list_experiments():
    """List all experiment files."""
    exp_path = get_experiments_path()
    return sorted(exp_path.glob("*.json"), reverse=True)


def list_maps():
    """List all custom map files."""
    maps_path = get_maps_path()
    return sorted(maps_path.glob("*.json"))
