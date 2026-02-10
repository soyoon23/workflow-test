"""Helpers for constructing LiteLLM-backed evaluator models for DeepEval tests."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml
from deepeval.models import LiteLLMModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@lru_cache(maxsize=1)
def load_project_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_eval_api_key(eval_cfg: Dict[str, Any]) -> str:
    api_key = os.environ.get("LLM_API_KEY", eval_cfg.get("api_key", ""))
    if api_key.startswith("${") and api_key.endswith("}"):
        env_name = api_key[2:-1]
        api_key = os.environ.get(env_name, "")
    return api_key


@lru_cache(maxsize=1)
def get_eval_model() -> LiteLLMModel:
    config = load_project_config()
    eval_cfg = config.get("eval", {})
    return LiteLLMModel(
        model=eval_cfg.get("model", "gpt-4o-mini"),
        base_url=eval_cfg.get("base_url", "https://api.openai.com/v1"),
        api_key=_resolve_eval_api_key(eval_cfg),
        temperature=eval_cfg.get("temperature", 0.0),
    )
