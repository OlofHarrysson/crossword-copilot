from joblib import Memory
from pathlib import Path

project_python_root = Path(__file__).parents[3]
err_msg = f"{project_python_root.name=} not in [{project_python_root.name}]"
assert project_python_root.name in ["backend"], err_msg
CACHE_DIR = f"{project_python_root}/data/cache"

Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
MEMORY = Memory(location=CACHE_DIR, verbose=0)
