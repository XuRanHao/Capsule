from capsule.enums import AssetType, EmbeddingType
from capsule.features import (
    asset_usage_embedding_text,
    effective_feature_text,
    embedding_channel_is_eligible,
    embedding_type_supports_any_asset_type,
    embedding_type_supports_asset_type,
)


def test_visual_dimensions_only_support_visual_asset_types() -> None:
    for embedding_type in (
        EmbeddingType.VISUAL_STYLE,
        EmbeddingType.COLOR_COMPOSITION,
    ):
        assert embedding_type_supports_asset_type(
            embedding_type=embedding_type,
            asset_type=AssetType.IMAGE,
        )
        assert embedding_type_supports_asset_type(
            embedding_type=embedding_type,
            asset_type=AssetType.VIDEO_SEGMENT,
        )
        assert not embedding_type_supports_asset_type(
            embedding_type=embedding_type,
            asset_type=AssetType.MARKDOWN_BLOCK,
        )
        assert not embedding_type_supports_asset_type(
            embedding_type=embedding_type,
            asset_type=AssetType.TEXT_BLOCK,
        )


def test_mixed_target_types_use_union_dimension_support() -> None:
    assert embedding_type_supports_any_asset_type(
        embedding_type=EmbeddingType.VISUAL_STYLE,
        asset_types=[AssetType.IMAGE, AssetType.MARKDOWN_BLOCK],
    )
    assert not embedding_type_supports_any_asset_type(
        embedding_type=EmbeddingType.VISUAL_STYLE,
        asset_types=[AssetType.MARKDOWN_BLOCK, AssetType.TEXT_BLOCK],
    )
    assert embedding_type_supports_any_asset_type(
        embedding_type=EmbeddingType.MOOD_ATMOSPHERE,
        asset_types=[AssetType.MARKDOWN_BLOCK, AssetType.TEXT_BLOCK],
    )


def test_null_feature_statuses_never_expose_stale_text() -> None:
    for status in ("unknown", "not_applicable"):
        features = {
            "character_state_or_psychology": {
                "value": "错误遗留的人物状态",
                "status": status,
            }
        }

        assert (
            effective_feature_text(
                features,
                EmbeddingType.CHARACTER_STATE_OR_PSYCHOLOGY,
            )
            is None
        )
        assert not embedding_channel_is_eligible(
            embedding_type=EmbeddingType.CHARACTER_STATE_OR_PSYCHOLOGY,
            asset_features=features,
        )


def test_explicit_null_effective_value_does_not_fall_back_to_model_value() -> None:
    features = {
        "visual_style": {
            "effective_value": None,
            "model_value": "赛博朋克",
            "status": "observed",
        }
    }

    assert effective_feature_text(features, EmbeddingType.VISUAL_STYLE) is None


def test_current_and_spec_feature_shapes_are_both_supported() -> None:
    current = {"visual_style": {"value": "赛博朋克", "status": "observed"}}
    specified = {
        "visual_style": {
            "model_value": "写实电影感",
            "user_value": "水彩",
            "status": "user_supplied",
        }
    }

    assert effective_feature_text(current, EmbeddingType.VISUAL_STYLE) == "赛博朋克"
    assert effective_feature_text(specified, EmbeddingType.VISUAL_STYLE) == "水彩"


def test_asset_usage_embedding_uses_directory_but_not_unique_filename() -> None:
    features = {
        "asset_usage": {
            "value": "海报制作",
            "status": "metadata",
            "source_path": "海报/素材/20251216-143446.png",
        }
    }

    text = asset_usage_embedding_text(
        features,
        "海报/素材/20251216-143446.png",
    )

    assert text == "素材用途：海报制作；来源目录：海报/素材"
    assert "20251216-143446.png" not in text
