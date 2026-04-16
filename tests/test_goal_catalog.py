from __future__ import annotations

from agent.embedding_provider import MockEmbeddingProvider
from agent.intent_index import get_intent_cards
from data.diagnostic_profiles import get_diagnostic_profiles, clear_diagnostic_profile_cache
from data.goal_catalog import build_goal_catalog


def test_goal_catalog_builds_goal_summary_from_csv(tmp_path, monkeypatch) -> None:
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
    clear_diagnostic_profile_cache()
    profiles = get_diagnostic_profiles()

    catalog = build_goal_catalog(profiles=profiles, cards=get_intent_cards(), provider=MockEmbeddingProvider())

    assert "VEHICLE_HEALTH_CHECK" in catalog
    entry = catalog["VEHICLE_HEALTH_CHECK"]
    assert entry.profile_count == 1
    assert "engine health" in " ".join(entry.symptoms).lower() or entry.symptoms
    assert "engine_rpm" in entry.requested_signal_keys


def test_goal_catalog_read_dtc_mapping_filters_non_dtc_profiles(tmp_path, monkeypatch) -> None:
    csv_path = tmp_path / "profiles.csv"
    csv_path.write_text(
        "\n".join(
            [
                "id,code,name,description,defaultRequestedPidsJson,includeDtcsByDefault,createdAt,updatedAt",
                '1,read_faults__show_stored_codes__constant,Read Faults,Read stored fault codes and DTCs.,"[]",True,2026-04-13,2026-04-13',
                '2,brake_noise__brakes_are_grinding__constant,Brake Noise,Brake grinding noise without DTC wording.,"[]",True,2026-04-13,2026-04-13',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DIAGNOSTIC_PROFILE_CSV_PATH", str(csv_path))
    clear_diagnostic_profile_cache()
    profiles = get_diagnostic_profiles()

    catalog = build_goal_catalog(profiles=profiles, cards=get_intent_cards(), provider=MockEmbeddingProvider())

    assert "READ_DTC" in catalog
    entry = catalog["READ_DTC"]
    assert entry.profile_count == 1
