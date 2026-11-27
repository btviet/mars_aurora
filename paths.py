import os
from pathlib import Path

def _fix_path(env_name: str) -> Path:
    _dir = os.getenv("VIRTUAL_ENV")
    if not _dir:
        raise OSError(f"Please update {env_name} in .env")
    _path = Path(_dir)
    if not _path.exists():
        raise FileNotFoundError(f"{_dir} do not exist")
    return _path


def get_project_root() -> Path:
    PROJECT_ROOT = Path(__file__).resolve().parent  # adjust .parent count to match where this file lives
    return PROJECT_ROOT
    #return _fix_path("LOCAL_WORKING_DIR")


def get_data_dir() -> Path:
    DATA_DIR = get_project_root() / "files/"
    return DATA_DIR


if __name__ == "__main__":
    print(get_data_dir())
    #_dir = os.getenv("VIRTUAL_ENV")
    #path = Path(_dir)
