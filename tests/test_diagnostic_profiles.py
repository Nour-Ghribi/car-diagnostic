from __future__ import annotations

from data.diagnostic_profiles import clear_diagnostic_profile_cache, get_diagnostic_profiles


def test_diagnostic_profile_loader_normalizes_csv_row(tmp_path, monkeypatch) -> None:
    csv_path = tmp_path / "profiles.csv"
    csv_path.write_text(
        "\n".join(
            [
                "id,code,name,description,defaultRequestedPidsJson,includeDtcsByDefault,createdAt,updatedAt",
                '1,cooling_overheating__car_overheats__under_load,Overheating,System: Cooling. Overheating profile.,"[{""key"": ""coolant_temp_c"", ""pid"": ""05"", ""mode"": ""01"", ""priority"": 1}]",True,2026-04-13,2026-04-13',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DIAGNOSTIC_PROFILE_CSV_PATH", str(csv_path))
    clear_diagnostic_profile_cache()

    profiles = get_diagnostic_profiles()

    assert len(profiles) == 1
    profile = profiles[0]
    assert profile.domain == "cooling overheating"
    assert profile.symptom == "car overheats"
    assert profile.context == "under load"
    assert profile.include_dtcs is True
    assert profile.requested_pids[0].key == "coolant_temp_c"

