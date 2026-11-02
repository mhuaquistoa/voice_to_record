"""Settings come from config.yaml; secrets come from .env. Nothing else is hard-coded."""
import json
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


@lru_cache(maxsize=1)
def load_config() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def path(key: str) -> Path:
    return ROOT / load_config()["paths"][key]


def ground_truth(task: str) -> dict:
    name = "interviews.json" if task == "interview" else "calls.json"
    data = json.loads((path("ground_truth") / name).read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def menu() -> dict:
    return json.loads((path("ground_truth") / "calls.json").read_text(encoding="utf-8"))["_menu"]
