"""Pytest configuration: insert the project root into sys.path.

This mirrors running pytest with ``PYTHONPATH=.`` so that imports like
``from models.video_mamba import ...`` resolve correctly from any test file.
"""

import sys
import os

# Project root is one level above the tests/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
