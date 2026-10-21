import importlib

SUBPACKAGES = ("data", "prompts", "training", "models", "decoding", "evaluation")


def test_import():
    importlib.import_module("thinkguard")
    for name in SUBPACKAGES:
        importlib.import_module(f"thinkguard.{name}")
