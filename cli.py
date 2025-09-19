#!/usr/bin/env python3
"""
CLI entry point for the PR Review Assistant
"""

import sys
import os

# Add the current directory to the path so we can import pr_reviewer
sys.path.insert(0, os.path.dirname(__file__))

from pr_reviewer.reviewer import main

if __name__ == "__main__":
    main()