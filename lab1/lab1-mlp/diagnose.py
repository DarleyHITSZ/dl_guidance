#!/usr/bin/env python3
"""
Diagnostic script to help identify why the MLP project won't run.
This script checks:
1. Python environment
2. Required dependencies
3. Project structure and files
4. Module imports
"""

import sys
import os
import subprocess
import importlib.util

def check_python_environment():
    """Check Python version and environment"""
    print("=== Python Environment Check ===")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"sys.path: {sys.path}")
    print()

def check_dependencies():
    """Check if required dependencies are installed"""
    print("=== Dependencies Check ===")
    requirements = [
        'numpy>=1.21',
        'pandas',
        'scikit-learn',
        'matplotlib'
    ]
    
    for req in requirements:
        pkg_name = req.split('>=')[0] if '>=' in req else req.split('<=')[0] if '<=' in req else req
        try:
            module = importlib.import_module(pkg_name)
            version = getattr(module, '__version__', 'Unknown version')
            print(f"✓ {pkg_name}: {version}")
        except ImportError:
            print(f"✗ {pkg_name}: NOT INSTALLED")
    print()

def check_project_structure():
    """Check if project files and directories exist"""
    print("=== Project Structure Check ===")
    expected_structure = [
        'demo.py',
        'demo_advanced.py',
        'requirements.txt',
        'README.md',
        'mlp/__init__.py',
        'mlp/layers.py',
        'mlp/activations.py',
        'mlp/losses.py',
        'mlp/optimizers.py',
        'mlp/model.py',
        'mlp/datasets.py'
    ]
    
    missing_files = []
    for path in expected_structure:
        if os.path.exists(path):
            print(f"✓ {path}")
        else:
            print(f"✗ {path}: MISSING")
            missing_files.append(path)
    
    if missing_files:
        print(f"\n{len(missing_files)} missing files found!")
    else:
        print("\nAll expected files are present.")
    print()
    return missing_files

def check_module_imports():
    """Test importing the mlp module and its components"""
    print("=== Module Imports Check ===")
    
    try:
        import mlp
        print("✓ Successfully imported mlp")
        
        # Test importing submodules
        from mlp import layers, activations, losses, optimizers, model, datasets
        print("✓ Successfully imported all submodules")
        
        # Test specific components
        from mlp.layers import Dense
        from mlp.activations import ReLU
        from mlp.losses import MSE
        from mlp.optimizers import Adam
        from mlp.model import MLP
        from mlp.datasets import BostonHousingLoader
        print("✓ Successfully imported all required components")
        
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error during import: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_simple_test():
    """Run a simple test to verify the MLP works"""
    print("\n=== Simple MLP Test ===")
    try:
        from mlp.model import MLP
        from mlp.layers import Dense
        from mlp.activations import ReLU, Linear
        from mlp.losses import MSE
        from mlp.optimizers import SGD
        
        # Create a simple model
        model = MLP(loss=MSE(), optimizer=SGD(learning_rate=0.01))
        model.add(Dense(2, 3))
        model.add(ReLU())
        model.add(Dense(3, 1))
        model.add(Linear())
        
        # Create dummy data
        import numpy as np
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])
        
        # Test a forward pass
        output = model.forward(X)
        print(f"✓ Simple forward pass successful")
        print(f"  Input shape: {X.shape}")
        print(f"  Output shape: {output.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Simple test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("MLP Project Diagnostic Tool")
    print("=" * 30)
    print()
    
    # Change to project root directory if not already there
    if not os.path.exists("mlp"):
        project_root = os.path.dirname(os.path.abspath(__file__))
        os.chdir(project_root)
        print(f"Changed to project root: {project_root}")
        print()
    
    check_python_environment()
    check_dependencies()
    missing_files = check_project_structure()
    import_success = check_module_imports()
    
    if import_success and not missing_files:
        run_simple_test()
        print("\n=== Diagnostic Summary ===")
        print("✅ All checks passed! The project should run correctly.")
        print("\nTo run the demo scripts:")
        print("  python demo.py")
        print("  python demo_advanced.py")
    else:
        print("\n=== Diagnostic Summary ===")
        print("❌ Some issues were found!")
        print("\nRecommended fixes:")
        if missing_files:
            print("  1. Restore missing files")
        print("  2. Install missing dependencies:")
        print("     python -m pip install -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com -r requirements.txt")
        print("  3. Make sure you're running from the project root directory")

if __name__ == "__main__":
    main()
