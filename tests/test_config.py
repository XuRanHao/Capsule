import pytest
from pydantic import ValidationError

from capsule.config import Settings


def test_blank_optional_secrets_are_unset() -> None:
    settings = Settings(ark_api_key="", milvus_token="")

    assert settings.ark_api_key is None
    assert settings.milvus_token is None


def test_document_chunk_size_defaults_to_250_400_500_600_tokens() -> None:
    settings = Settings(_env_file=None)

    assert settings.assetization_version == "assetization-v6"
    assert settings.search_weight_model == "doubao-seed-2-0-mini-260428"
    assert settings.search_weight_max_output_tokens == 128
    assert settings.document_tokenizer_path is None
    assert settings.document_chunk_min_tokens == 250
    assert settings.document_chunk_target_tokens == 400
    assert settings.document_chunk_max_tokens == 500
    assert settings.document_chunk_merge_max_tokens == 600


def test_legacy_search_parser_environment_names_remain_compatible(monkeypatch) -> None:
    monkeypatch.delenv("CAPSULE_SEARCH_WEIGHT_MODEL", raising=False)
    monkeypatch.delenv("CAPSULE_SEARCH_WEIGHT_MAX_OUTPUT_TOKENS", raising=False)
    monkeypatch.setenv("CAPSULE_SEARCH_PARSER_MODEL", "legacy-parser-model")
    monkeypatch.setenv("CAPSULE_SEARCH_PARSER_MAX_OUTPUT_TOKENS", "256")

    settings = Settings(_env_file=None)

    assert settings.search_weight_model == "legacy-parser-model"
    assert settings.search_weight_max_output_tokens == 256


def test_search_weight_environment_names_override_legacy_aliases(monkeypatch) -> None:
    monkeypatch.setenv("CAPSULE_SEARCH_WEIGHT_MODEL", "weight-model")
    monkeypatch.setenv("CAPSULE_SEARCH_WEIGHT_MAX_OUTPUT_TOKENS", "192")
    monkeypatch.setenv("CAPSULE_SEARCH_PARSER_MODEL", "legacy-parser-model")
    monkeypatch.setenv("CAPSULE_SEARCH_PARSER_MAX_OUTPUT_TOKENS", "256")

    settings = Settings(_env_file=None)

    assert settings.search_weight_model == "weight-model"
    assert settings.search_weight_max_output_tokens == 192


def test_video_adaptive_segmentation_defaults_replace_legacy_settings() -> None:
    settings = Settings()

    assert settings.assetization_version == "assetization-v6"
    assert settings.video_distance_quantile == 0.75
    assert settings.video_activity_sample_fps == 6.0
    assert settings.video_keyframe_jpeg_quality == 85
    assert "video_scene_threshold" not in Settings.model_fields
    assert "video_max_candidate_frames" not in Settings.model_fields


def test_video_keyframe_size_cannot_diverge_from_media_writer_contract() -> None:
    with pytest.raises(ValidationError):
        Settings(video_keyframe_size=256)


def test_incremental_cluster_defaults_are_conservative() -> None:
    settings = Settings()

    assert settings.cluster_incremental_assignment_threshold == 0.92
    assert settings.cluster_recluster_ratio_threshold == 0.10
    assert settings.cluster_recluster_minimum_count == 50
    assert settings.cluster_recluster_concurrency == 1
