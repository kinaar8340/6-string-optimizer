"""Fisher-Rao benchmark: symlink to ~/Projects/Fisher_Rao/examples/rosenbrock_fr_benchmark.py"""

import runpy
import sys
from pathlib import Path

EXAMPLE = Path.home() / "Projects" / "Fisher_Rao" / "examples" / "rosenbrock_fr_benchmark.py"
if not EXAMPLE.exists():
    raise FileNotFoundError(f"Fisher-Rao example not found: {EXAMPLE}")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
runpy.run_path(str(EXAMPLE), run_name="__main__")