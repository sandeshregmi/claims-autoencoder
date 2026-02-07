#!/usr/bin/env python3
"""
Databricks Bundle Validation Script
Checks that all components are properly configured before deployment
"""

import os
import sys
import yaml
import subprocess
from pathlib import Path
from typing import List, Tuple


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")


def print_success(text: str):
    print(f"{Colors.GREEN}✓{Colors.RESET} {text}")


def print_error(text: str):
    print(f"{Colors.RED}✗{Colors.RESET} {text}")


def print_warning(text: str):
    print(f"{Colors.YELLOW}⚠{Colors.RESET} {text}")


def print_info(text: str):
    print(f"{Colors.BLUE}ℹ{Colors.RESET} {text}")


def check_file_exists(filepath: str) -> bool:
    """Check if a file exists"""
    return Path(filepath).exists()


def check_databricks_cli() -> bool:
    """Check if Databricks CLI is installed"""
    try:
        result = subprocess.run(
            ['databricks', '--version'],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def check_databricks_config() -> Tuple[bool, str]:
    """Check if Databricks CLI is configured"""
    config_path = Path.home() / '.databrickscfg'
    
    if not config_path.exists():
        return False, "Config file not found"
    
    try:
        with open(config_path, 'r') as f:
            content = f.read()
            if 'host' in content and 'token' in content:
                return True, "Configured"
            else:
                return False, "Missing host or token"
    except Exception as e:
        return False, str(e)


def validate_yaml(filepath: str) -> Tuple[bool, str]:
    """Validate YAML file syntax"""
    try:
        with open(filepath, 'r') as f:
            yaml.safe_load(f)
        return True, "Valid YAML"
    except yaml.YAMLError as e:
        return False, str(e)
    except FileNotFoundError:
        return False, "File not found"


def check_required_files() -> List[Tuple[str, bool]]:
    """Check if all required files exist"""
    required_files = [
        'databricks.yml',
        'resources/jobs/training_job.yml',
        'resources/jobs/scoring_job.yml',
        'resources/jobs/monitoring_job.yml',
        'src/data_ingestion.py',
        'src/tree_models.py',
        'src/fairness_analysis.py',
        'src/psi_monitoring.py',
        'requirements.txt'
    ]
    
    results = []
    for filepath in required_files:
        exists = check_file_exists(filepath)
        results.append((filepath, exists))
    
    return results


def check_python_modules():
    """Check if required Python modules are available"""
    required_modules = [
        'pandas',
        'numpy',
        'scipy',
        'catboost',
        'xgboost',
        'mlflow'
    ]
    
    results = []
    for module in required_modules:
        try:
            __import__(module)
            results.append((module, True))
        except ImportError:
            results.append((module, False))
    
    return results


def run_bundle_validate() -> Tuple[bool, str]:
    """Run databricks bundle validate"""
    try:
        result = subprocess.run(
            ['databricks', 'bundle', 'validate'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            return True, "Bundle is valid"
        else:
            return False, result.stderr or result.stdout
    except subprocess.TimeoutExpired:
        return False, "Validation timed out"
    except FileNotFoundError:
        return False, "Databricks CLI not found"
    except Exception as e:
        return False, str(e)


def main():
    print_header("Databricks Bundle Validation")
    
    # Track overall status
    all_passed = True
    
    # 1. Check Databricks CLI
    print_header("1. Databricks CLI")
    if check_databricks_cli():
        print_success("Databricks CLI is installed")
        
        # Check version
        result = subprocess.run(
            ['databricks', '--version'],
            capture_output=True,
            text=True
        )
        print_info(f"Version: {result.stdout.strip()}")
    else:
        print_error("Databricks CLI is not installed")
        print_info("Install with: pip install databricks-cli")
        all_passed = False
    
    # 2. Check Databricks configuration
    print_header("2. Databricks Configuration")
    is_configured, message = check_databricks_config()
    if is_configured:
        print_success(f"Databricks CLI is configured: {message}")
    else:
        print_error(f"Databricks CLI not configured: {message}")
        print_info("Configure with: databricks configure --token")
        all_passed = False
    
    # 3. Check required files
    print_header("3. Required Files")
    file_results = check_required_files()
    
    missing_files = []
    for filepath, exists in file_results:
        if exists:
            print_success(f"{filepath}")
        else:
            print_error(f"{filepath} - MISSING")
            missing_files.append(filepath)
            all_passed = False
    
    if missing_files:
        print_warning(f"\n{len(missing_files)} required files are missing")
    
    # 4. Validate YAML files
    print_header("4. YAML Validation")
    
    yaml_files = [
        'databricks.yml',
        'resources/jobs/training_job.yml',
        'resources/jobs/scoring_job.yml',
        'resources/jobs/monitoring_job.yml'
    ]
    
    for filepath in yaml_files:
        if check_file_exists(filepath):
            is_valid, message = validate_yaml(filepath)
            if is_valid:
                print_success(f"{filepath}: {message}")
            else:
                print_error(f"{filepath}: {message}")
                all_passed = False
        else:
            print_warning(f"{filepath}: File not found")
    
    # 5. Check Python modules
    print_header("5. Python Dependencies")
    module_results = check_python_modules()
    
    missing_modules = []
    for module, available in module_results:
        if available:
            print_success(f"{module}")
        else:
            print_warning(f"{module} - Not installed")
            missing_modules.append(module)
    
    if missing_modules:
        print_info(f"\nOptional: Install with: pip install {' '.join(missing_modules)}")
    
    # 6. Run bundle validate
    print_header("6. Bundle Validation")
    if check_databricks_cli() and is_configured:
        print_info("Running: databricks bundle validate")
        is_valid, message = run_bundle_validate()
        
        if is_valid:
            print_success(f"Bundle validation passed: {message}")
        else:
            print_error(f"Bundle validation failed:")
            print(f"\n{message}\n")
            all_passed = False
    else:
        print_warning("Skipping bundle validation (CLI not configured)")
    
    # 7. Summary
    print_header("Summary")
    
    if all_passed:
        print_success("All checks passed! ✓")
        print_info("\nReady to deploy:")
        print("  databricks bundle deploy --target dev")
        return 0
    else:
        print_error("Some checks failed ✗")
        print_info("\nPlease fix the issues above before deploying")
        return 1


if __name__ == "__main__":
    sys.exit(main())
