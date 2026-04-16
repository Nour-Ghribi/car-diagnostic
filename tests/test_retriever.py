from __future__ import annotations

from agent.embedding_provider import clear_embedding_cache
from agent.settings import clear_settings_cache
from data.diagnostic_profiles import clear_diagnostic_profile_cache
from nlp.retriever import retrieve_candidates


def test_retriever_returns_vehicle_health_candidate_for_broad_prompt(monkeypatch) -> None:
    monkeypatch.setenv("INTENT_EMBEDDING_PROVIDER", "mock")
    clear_settings_cache()
    clear_embedding_cache()
    clear_diagnostic_profile_cache()
    candidates = retrieve_candidates("je veux connaitre le health de ma voiture")
    assert candidates
    assert candidates[0].goal == "VEHICLE_HEALTH_CHECK"
    assert 0.0 <= candidates[0].score <= 1.0


def test_retriever_enriches_candidates_with_supporting_profiles(tmp_path, monkeypatch) -> None:
    csv_path = tmp_path / "profiles.csv"
    csv_path.write_text(
        "\n".join(
            [
                "id,code,name,description,defaultRequestedPidsJson,includeDtcsByDefault,createdAt,updatedAt",
                '1,engine_health__engine_is_healthy__overall,Engine Health,System: Engine performance. User wants overall engine health.,"[{""key"": ""engine_rpm"", ""pid"": ""0C"", ""mode"": ""01"", ""priority"": 1}, {""key"": ""coolant_temp_c"", ""pid"": ""05"", ""mode"": ""01"", ""priority"": 2}]",True,2026-04-13,2026-04-13',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DIAGNOSTIC_PROFILE_CSV_PATH", str(csv_path))
    monkeypatch.setenv("INTENT_EMBEDDING_PROVIDER", "mock")
    clear_settings_cache()
    clear_embedding_cache()
    clear_diagnostic_profile_cache()

    candidates = retrieve_candidates("check my car health")

    assert candidates
    top = candidates[0]
    assert top.goal == "VEHICLE_HEALTH_CHECK"
    assert "supporting_profiles" in top.metadata
    assert top.metadata["supporting_profiles"]
    assert "supporting_profiles_global" in top.metadata
    assert "goal_profile_summary" in top.metadata


def test_retriever_boosts_vehicle_context_lookup_for_explicit_vin_prompt(monkeypatch) -> None:
    monkeypatch.setenv("INTENT_EMBEDDING_PROVIDER", "mock")
    clear_settings_cache()
    clear_embedding_cache()
    clear_diagnostic_profile_cache()

    candidates = retrieve_candidates("get vin")

    assert candidates
    assert candidates[0].goal == "VEHICLE_CONTEXT_LOOKUP"
