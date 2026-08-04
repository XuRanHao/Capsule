import pytest
from pydantic import ValidationError

from capsule.config import Settings


def test_blank_optional_secrets_are_unset() -> None:
    settings = Settings(ark_api_key="", milvus_token="")

    assert settings.ark_api_key is None
    assert settings.milvus_token is None


def test_document_chunk_size_defaults_to_250_400_500_600_tokens() -> None:
    settings = Settings()

    assert settings.assetization_version == "assetization-v6"
    assert settings.search_parser_model == "doubao-seed-2-0-mini-260428"
    assert settings.document_tokenizer_path is None
    assert settings.document_chunk_min_tokens == 250
    assert settings.document_chunk_target_tokens == 400
    assert settings.document_chunk_max_tokens == 500
    assert settings.document_chunk_merge_max_tokens == 600


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
