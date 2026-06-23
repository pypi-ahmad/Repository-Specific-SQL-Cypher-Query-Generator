import pytest

from repo_query_gen.config import Settings
from repo_query_gen.training import _resolve_backend


def test_unsloth_incompatible_model_falls_back_to_trl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "repo_query_gen.training._is_module_available",
        lambda module: module in {"unsloth", "trl"},
    )
    monkeypatch.setattr("repo_query_gen.training._unsloth_model_supported", lambda _: False)

    resolution = _resolve_backend("unsloth", Settings(), allow_fallback=True)
    assert resolution.effective == "trl"
    assert "unsloth_model_incompatible" in str(resolution.fallback_reason)


def test_trl_missing_raises_when_fallback_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("repo_query_gen.training._is_module_available", lambda _: False)

    with pytest.raises(RuntimeError):
        _resolve_backend("trl", Settings(), allow_fallback=False)
