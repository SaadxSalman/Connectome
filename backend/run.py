"""Launch the SynapseCraft neural gateway (uvicorn)."""

import sys
import uvicorn

from app.config import Settings

# Windows consoles default to a legacy code page that chokes on glyphs.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

if __name__ == "__main__":
    settings = Settings()
    print(f"⚡ SynapseCraft gateway → http://{settings.app_host}:{settings.app_port}")
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False,
        log_level="info",
    )
