import importlib.util
import sys
from pathlib import Path

PLATFORM_ROOT = Path(__file__).resolve().parent / "rag-platform"
API_ROOT = PLATFORM_ROOT / "api"
API_MAIN = API_ROOT / "main.py"
for path in (str(PLATFORM_ROOT), str(API_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

spec = importlib.util.spec_from_file_location("rag_platform_api_main", API_MAIN)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
app = module.app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
