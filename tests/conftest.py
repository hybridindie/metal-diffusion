import sys
import os
import pytest

# Add src to sys.path so tests can import alloy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
