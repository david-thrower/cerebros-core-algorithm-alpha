# Ensure the repository root is on sys.path so `import cerebros` works when
# running tests without installing the package.
from os.path import abspath, dirname, join
from sys import path

ROOT = abspath(join(dirname(__file__), ".."))
if ROOT not in path:
    path.insert(0, ROOT)
