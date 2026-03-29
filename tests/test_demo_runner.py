from __future__ import annotations

from io import StringIO

from agent.settings import clear_settings_cache
from demo.run_real_mode_demo import run_demo_once


def test_demo_runner_prints_structured_response(monkeypatch) -> None:
    monkeypatch.setenv("INTENT_EMBEDDING_PROVIDER", "mock")
    monkeypatch.setenv("INTENT_RERANKER_PROVIDER", "mock")
    clear_settings_cache()

    output = StringIO()
    run_demo_once("veh_002", "je veux connaître l'état du moteur", out_stream=output)
    rendered = output.getvalue()

    assert "Real Mode Demo" in rendered
    assert "OBD/data layer     : mock-backed tools" in rendered
    assert "Intent" in rendered
    assert "Diagnostic" in rendered
    assert "Recommendations" in rendered

    clear_settings_cache()
