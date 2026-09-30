import os
import sys

# metrics.py is a flat module under src/, imported the same way evaluate.py
# imports it (sys.path insertion, not a package) — match that convention here
# rather than introducing a different import style just for tests.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
