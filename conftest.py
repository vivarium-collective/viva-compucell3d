"""Make the workspace package importable during tests.

cc3d is conda-only and pins Python 3.10, but the package declares
``requires-python >=3.11``, so it can't be editable-installed into the cc3d
(pixi) env. Putting the repo root on ``sys.path`` lets ``pytest`` import
``pbg_compucell3d`` from the source tree in that env. CI's uv (3.12) env
installs the package normally, where this insert is a harmless no-op.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
