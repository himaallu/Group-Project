"""generate._load reuses the claim extractor's Qwen3-8B copy instead of loading a second one."""
import sys
import types

from src.explain import generate


def test_load_reuses_extractor_model(monkeypatch):
    fake = types.ModuleType("src.claims.extract")
    fake.MODEL = generate.MODEL
    fake._model = ("tok", "model")
    monkeypatch.setitem(sys.modules, "src.claims.extract", fake)
    monkeypatch.setattr(generate, "_model", None)
    assert generate._load() == ("tok", "model")
