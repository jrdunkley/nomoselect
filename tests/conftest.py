"""Test configuration — set up import paths.

Works in three scenarios:
1. Both packages installed via pip (editable or not) — no path munging needed.
2. Running from the workspace with packages side-by-side — adds src dirs.
3. Sandbox environment (session paths) — same as (2) with absolute paths.
"""
import sys
import os

# Walk up from this file to the workspace root (nomoselect/../)
_this_dir = os.path.dirname(os.path.abspath(__file__))
_nomoselect_root = os.path.dirname(_this_dir)
_workspace = os.path.dirname(_nomoselect_root)

# Add src directories if they exist and are not already importable
_candidates = [
    os.path.join(_workspace, "observer_geometry", "src"),  # nomogeo
    os.path.join(_nomoselect_root, "src"),                  # nomoselect
]

for path in _candidates:
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)
