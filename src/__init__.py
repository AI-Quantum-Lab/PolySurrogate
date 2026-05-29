"""
Automatic path setup for the repository.

This allows the project to keep source files in `src/` and config files in
`configs/`, while still supporting simple internal imports such as:

    from data_generation_utils import ...
    from data_config import ...
    from models import ...

Run scripts from the repository root using:

    python -m src.data_generation
    python -m src.model_training
    python -m src.optimiser
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
CONFIG_DIR = REPO_ROOT / "configs"

for path in [str(SRC_DIR), str(CONFIG_DIR), str(REPO_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)