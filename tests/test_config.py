import pytest
from pydantic import ValidationError

from capsule.config import Settings


def test_blank_optional_secrets_are_unset() -> None:
    settings = Settings(ark_api_key="", deepseek_api_key="", milvus_token="")

    assert settings.ark_api_key is None
    assert settings.deepseek_api_key is None
    assert settings.milvus_token is None


def test_document_chunk_size_defaults_to_250_400_500_600_tokens() -> None:
    settings = Settings(_env_file=None)

    assert settings.assetization_version == "assetization-v6"
    assert settings.deepseek_base_url == "https://api.deepseek.com"
    assert settings.search_query_model == "deepseek-v4-flash"
    assert settings.search_query_max_output_tokens == 500
    assert settings.document_tokenizer_path is None
    assert settings.document_chunk_min_tokens == 250
    assert settings.document_chunk_target_tokens == 400
    assert settings.document_chunk_max_tokens == 500
    assert settings.document_chunk_merge_max_tokens == 600


def test_legacy_search_parser_environment_names_remain_compatible(monkeypatch) -> None:
    monkeypatch.delenv("CAPSULE_SEARCH_QUERY_MODEL", raising=False)
    monkeypatch.delenv("CAPSULE_SEARCH_QUERY_MAX_OUTPUT_TOKENS", raising=False)
    monkeypatch.delenv("CAPSULE_SEARCH_WEIGHT_MODEL", raising=False)
    monkeypatch.delenv("CAPSULE_SEARCH_WEIGHT_MAX_OUTPUT_TOKENS", raising=False)
    monkeypatch.setenv("CAPSULE_SEARCH_PARSER_MODEL", "legacy-parser-model")
    monkeypatch.setenv("CAPSULE_SEARCH_PARSER_MAX_OUTPUT_TOKENS", "256")

    settings = Settings(_env_file=None)

    assert settings.search_query_model == "legacy-parser-model"
    assert settings.search_query_max_output_tokens == 256


def test_legacy_search_weight_environment_names_remain_compatible(monkeypatch) -> None:
    monkeypatch.delenv("CAPSULE_SEARCH_QUERY_MODEL", raising=False)
    monkeypatch.delenv("CAPSULE_SEARCH_QUERY_MAX_OUTPUT_TOKENS", raising=False)
    monkeypatch.setenv("CAPSULE_SEARCH_WEIGHT_MODEL", "weight-model")
    monkeypatch.setenv("CAPSULE_SEARCH_WEIGHT_MAX_OUTPUT_TOKENS", "192")
    monkeypatch.setenv("CAPSULE_SEARCH_PARSER_MODEL", "legacy-parser-model")
    monkeypatch.setenv("CAPSULE_SEARCH_PARSER_MAX_OUTPUT_TOKENS", "256")

    settings = Settings(_env_file=None)

    assert settings.search_query_model == "weight-model"
    assert settings.search_query_max_output_tokens == 192


def test_search_query_environment_names_override_legacy_aliases(monkeypatch) -> None:
    monkeypatch.setenv("CAPSULE_SEARCH_QUERY_MODEL", "query-model")
    monkeypatch.setenv("CAPSULE_SEARCH_QUERY_MAX_OUTPUT_TOKENS", "500")
    monkeypatch.setenv("CAPSULE_SEARCH_WEIGHT_MODEL", "weight-model")
    monkeypatch.setenv("CAPSULE_SEARCH_WEIGHT_MAX_OUTPUT_TOKENS", "192")
    monkeypatch.setenv("CAPSULE_SEARCH_PARSER_MODEL", "legacy-parser-model")
    monkeypatch.setenv("CAPSULE_SEARCH_PARSER_MAX_OUTPUT_TOKENS", "256")

    settings = Settings(_env_file=None)

    assert settings.search_query_model == "query-model"
    assert settings.search_query_max_output_tokens == 500


def test_video_adaptive_segmentation_defaults_replace_legacy_settings() -> None:
    settings = Settings(_env_file=None)

    assert settings.assetization_version == "assetization-v6"
    assert settings.ffmpeg_concurrency == 2
    assert settings.video_distance_quantile == 0.75
    assert settings.video_output_mode == "logical"
    assert settings.video_activity_sample_fps == 6.0
    assert settings.video_keyframe_jpeg_quality == 85
    assert settings.understanding_image_size == 768
    assert settings.api_embedded_cpu_tasks_enabled
    assert "video_scene_threshold" not in Settings.model_fields
    assert "video_max_candidate_frames" not in Settings.model_fields


def test_video_keyframe_size_cannot_diverge_from_media_writer_contract() -> None:
    with pytest.raises(ValidationError):
        Settings(video_keyframe_size=256)


def test_video_output_mode_can_preserve_materialized_segment_compatibility() -> None:
    assert Settings(video_output_mode="materialized").video_output_mode == "materialized"


def test_incremental_cluster_defaults_balance_recall_and_precision() -> None:
    settings = Settings()

    assert settings.cluster_incremental_assignment_threshold == 0.88
    assert settings.cluster_bootstrap_minimum_count == 50
    assert settings.cluster_bootstrap_concurrency == 1
    assert settings.cluster_auto_recluster_new_ratio == 0.3
    assert settings.cluster_auto_recluster_minimum_new_count == 20
    assert "cluster_recluster_ratio_threshold" not in Settings.model_fields
    assert "cluster_recluster_minimum_count" not in Settings.model_fields
