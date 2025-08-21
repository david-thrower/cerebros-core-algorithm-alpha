# Ensure the repository root is on sys.path so `import cerebros` works when
# running tests without installing the package.
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
