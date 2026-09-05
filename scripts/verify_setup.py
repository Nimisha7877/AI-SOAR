"""Environment / install verification for AI SOAR.

Run after installing to confirm the package, dependencies, config and
directory layout are all healthy:

    python scripts/verify_setup.py

Exits 0 on success, 1 if a required dependency is missing.
"""

from __future__ import annotations

import importlib.metadata as md
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # allow running before `pip install -e .`

REQUIRED = [
    "pandas",
    "numpy",
    "pyarrow",
    "scikit-learn",
    "lightgbm",
    "shap",
    "pydantic",
    "PyYAML",
    "python-dotenv",
    "fastapi",
    "uvicorn",
    "httpx",
    "matplotlib",
    "seaborn",
]


def _version(dist: str) -> str:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return "MISSING"


def main() -> int:
    from ai_soar import __version__
    from ai_soar.config import get_settings
    from ai_soar.utils.logging import configure_from_settings, get_logger

    settings = get_settings()
    configure_from_settings(settings)
    log = get_logger("verify_setup")

    print(f"\nAI SOAR v{__version__}  -  setup verification")
    print(f"project root : {ROOT}")
    print(f"python       : {sys.version.split()[0]}\n")

    print("-- dependencies " + "-" * 46)
    missing = []
    for dist in REQUIRED:
        v = _version(dist)
        flag = "ok " if v != "MISSING" else "!! "
        if v == "MISSING":
            missing.append(dist)
        print(f"  [{flag}] {dist:<16} {v}")

    print("\n-- resolved paths " + "-" * 44)
    for name, p in settings.paths.model_dump().items():
        exists = Path(p).exists()
        print(f"  [{'ok ' if exists else '!! '}] {name:<15} {p}")

    print("\n-- config " + "-" * 52)
    print(f"  llm      : {settings.llm.provider}/{settings.llm.model}")
    print(f"  api      : {settings.api.host}:{settings.api.port}")
    print(f"  n8n      : enabled={settings.n8n.enabled} url={settings.n8n.webhook_url}")
    print(f"  response : backend={settings.response.backend}")
    print(f"  log      : level={settings.logging.level}")

    settings.ensure_directories()
    log.info("Directory layout ensured.")

    print()
    if missing:
        print(f"FAILED - missing dependencies: {', '.join(missing)}")
        print("Install with:  pip install -r requirements.txt   (or  pip install -e '.[dev]')")
        return 1
    print("ALL CHECKS PASSED - ready for step 3 (data pipeline).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())