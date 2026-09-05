#!/usr/bin/env python
"""Start the AI SOAR inference API.

Run:
    python scripts/serve_api.py                 # host/port from config/settings.yaml
    python scripts/serve_api.py --port 8001     # override port
    python scripts/serve_api.py --reload        # dev mode: reload on code change

Then, in a SECOND terminal:
    Invoke-RestMethod http://127.0.0.1:8000/health | ConvertTo-Json -Depth 4

Interactive docs (Swagger UI):  http://127.0.0.1:8000/docs

Pre-flight behaviour: if the trained models are missing, this script says so
plainly and exits BEFORE uvicorn starts, instead of letting the API come up
half-alive and answer every request with a 500.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

STAGE1 = "binary_stage1.joblib"
STAGE2 = "multiclass_stage2.joblib"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the AI SOAR inference API")
    parser.add_argument("--host", default=None, help="bind address (default: settings.api.host)")
    parser.add_argument("--port", type=int, default=None, help="port (default: settings.api.port)")
    parser.add_argument("--reload", action="store_true", help="auto-reload on code changes (dev)")
    parser.add_argument("--log-level", default="info", choices=["critical", "error", "warning", "info", "debug"])
    return parser.parse_args(argv)


def preflight(models_dir: Path) -> bool:
    """True if both trained models exist."""
    missing = [name for name in (STAGE1, STAGE2) if not (models_dir / name).exists()]
    if missing:
        print("\n[pre-flight] FAILED - trained model(s) missing:")
        for name in missing:
            print(f"  - {models_dir / name}")
        print("\nFix: run the training step first")
        print('  python scripts/train_models.py')
        print(f"(expected models directory: {models_dir})\n")
        return False
    print(f"[pre-flight] models found in {models_dir}")
    return True


def main(argv: list[str] | None = None) -> int:
    import uvicorn  # imported late: pre-flight errors stay readable

    from ai_soar.config import get_settings
    from ai_soar.inference.predictor import AUTO_RESPONSE_FAMILIES, DEFAULT_CONFIDENCE_FLOOR
    from ai_soar.utils.logging import configure_from_settings

    args = parse_args(argv)
    settings = get_settings()
    configure_from_settings(settings)
    settings.ensure_directories()

    host = args.host or settings.api.host
    port = args.port or settings.api.port
    models_dir = Path(settings.paths.models)

    if not preflight(models_dir):
        return 1

    browse_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    print("\n================ AI SOAR INFERENCE API ================")
    print(f"  host:port      : {host}:{port}")
    print(f"  docs (Swagger) : http://{browse_host}:{port}/docs")
    print(f"  health         : http://{browse_host}:{port}/health")
    print(f"  predict        : POST http://{browse_host}:{port}/predict")
    print(f"  policy         : gate=0.5  confidence_floor={DEFAULT_CONFIDENCE_FLOOR}")
    print(f"  auto-response  : {sorted(AUTO_RESPONSE_FAMILIES)}")
    print("  other families : human_approval (see leakage audit report)")
    print("  stop           : Ctrl+C")
    print("=======================================================\n")

    uvicorn.run(
        "ai_soar.inference.api:app",
        host=host,
        port=port,
        reload=args.reload,
        log_level=args.log_level,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())