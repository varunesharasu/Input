#!/usr/bin/env python3
"""
Quick fix script for path issues
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

def fix_traditional_models():
    """Fix the traditional_models.py file"""
    file_path = Path(__file__).parent.parent / "src" / "models" / "traditional_models.py"
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix the imports
    if "from config.config import PATHS" in content and "MODELS_DIR" not in content:
        content = content.replace(
            "from config.config import PATHS",
            "from config.config import PATHS, MODELS_DIR"
        )
    
    # Fix the save_models method
    content = content.replace(
        "model_path = PATHS['models_dir'] / f\"{name}_model.pkl\"",
        "model_path = MODELS_DIR / f\"{name}_model.pkl\""
    )
    
    # Fix the load_models method
    content = content.replace(
        "model_path = PATHS['models_dir'] / f\"{name}_model.pkl\"",
        "model_path = MODELS_DIR / f\"{name}_model.pkl\""
    )
    
    # Write back
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("Fixed traditional_models.py")

def fix_deep_learning_models():
    """Fix the deep_learning_models.py file"""
    file_path = Path(__file__).parent.parent / "src" / "models" / "deep_learning_models.py"
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix the imports
    if "from config.config import MODEL_CONFIG, PATHS" in content and "MODELS_DIR" not in content:
        content = content.replace(
            "from config.config import MODEL_CONFIG, PATHS",
            "from config.config import MODEL_CONFIG, PATHS, MODELS_DIR"
        )
    
    # Fix model checkpoint paths
    content = content.replace(
        "str(PATHS['models_dir'] / f\"{model_name}_model.h5\")",
        "str(MODELS_DIR / f\"{model_name}_model.h5\")"
    )
    
    # Fix tokenizer save path
    content = content.replace(
        "with open(PATHS['models_dir'] / 'tokenizer.pkl', 'wb') as f:",
        "with open(MODELS_DIR / 'tokenizer.pkl', 'wb') as f:"
    )
    
    # Fix tokenizer load path
    content = content.replace(
        "with open(PATHS['models_dir'] / 'tokenizer.pkl', 'rb') as f:",
        "with open(MODELS_DIR / 'tokenizer.pkl', 'rb') as f:"
    )
    
    # Fix model load path
    content = content.replace(
        "model_path = PATHS['models_dir'] / f\"{model_name}_model.h5\"",
        "model_path = MODELS_DIR / f\"{model_name}_model.h5\""
    )
    
    # Write back
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("Fixed deep_learning_models.py")

def fix_config():
    """Fix the config.py file"""
    file_path = Path(__file__).parent.parent / "config" / "config.py"
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Add missing keys to PATHS
    if "'models_dir': MODELS_DIR" not in content:
        content = content.replace(
            "'ensemble_model': MODELS_DIR / \"ensemble_model.pkl\"",
            "'ensemble_model': MODELS_DIR / \"ensemble_model.pkl\",\n    'models_dir': MODELS_DIR,\n    'results_dir': RESULTS_DIR"
        )
    
    # Write back
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("Fixed config.py")

def main():
    """Run all fixes"""
    print("Applying path fixes...")
    
    try:
        fix_config()
        fix_traditional_models()
        fix_deep_learning_models()
        print("All fixes applied successfully!")
        print("You can now run: python scripts/train_models.py")
    except Exception as e:
        print(f"Error applying fixes: {e}")

if __name__ == "__main__":
    main()
