"""
tests/conftest.py

Adds the project root to sys.path so that 'src' and 'api' imports
work in all test files without installing the package.
"""
import sys
from pathlib import Path

# Project root = one level above tests/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
