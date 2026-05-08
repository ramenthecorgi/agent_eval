import os
import sys
from pathlib import Path

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-placeholder")
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
