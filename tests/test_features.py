from capsule.enums import AssetType, EmbeddingType
from capsule.features import (
    FEATURE_DIMENSION_DISAMBIGUATION_PROMPT,
    FEATURE_DIMENSION_SCOPES,
    FEATURE_EMBEDDING_TYPES,
    effective_feature_text,
    embedding_channel_is_eligible,
    embedding_type_supports_any_asset_type,
    embedding_type_supports_asset_type,
)
from capsule.schemas import AssetFeatures


def test_positive_dimension_scopes_cover_all_feature_embedding_types() -> None:
    assert set(FEATURE_DIMENSION_SCOPES) == set(FEATURE_EMBEDDING_TYPES)
    subject_scope = FEATURE_DIMENSION_SCOPES[EmbeddingType.SUBJECT_CONTENT]
    scene_scope = FEATURE_DIMENSION_SCOPES[EmbeddingType.SCENE_THEME]
    visual_scope = FEATURE_DIMENSION_SCOPES[EmbeddingType.VISUAL_PRESENTATION]

    assert "可被指认的核心对象及其内容事实" in subject_scope
    assert "不同背景和视觉风格中找到同类主体" in subject_scope
    assert "整幅素材所呈现的全局情境" in scene_scope
    assert "找到同类场所、活动或故事情境" in scene_scope
    assert "素材如何被视觉化呈现" in visual_scope
    assert "跨主体、跨场景找到呈现方式相近" in visual_scope
    assert "主辅色" in visual_scope
    assert "光线方向与软硬" in visual_scope
    assert "前中后景" in visual_scope
    assert "主体、背景、前景、点缀" in visual_scope

    assert "三个互补的检索视角" in FEATURE_DIMENSION_DISAMBIGUATION_PROMPT
    assert "分别完成三次聚焦" in FEATURE_DIMENSION_DISAMBIGUATION_PROMPT
    assert FEATURE_DIMENSION_DISAMBIGUATION_PROMPT.count("不要把") == 4


def test_asset_feature_schema_uses_dimension_specific_positive_scopes() -> None:
    schema = str(AssetFeatures.model_json_schema())

    assert "主体 + 维度信息" not in schema
    assert "可被指认的核心对象及其内容事实" in schema
    assert "整幅素材所呈现的全局情境" in schema
    assert "素材如何被视觉化呈现" in schema
    assert "渲染质感" in schema
    assert "视觉重心" in schema
    assert "跨主体、跨场景找到呈现方式相近" in schema
    assert "不能形成独立全局情境" in schema
    assert "情绪氛围" not in schema
    assert "目标受众" not in schema


def test_visual_dimensions_only_support_visual_asset_types() -> None:
    for embedding_type in (EmbeddingType.VISUAL_PRESENTATION,):
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
        embedding_type=EmbeddingType.VISUAL_PRESENTATION,
        asset_types=[AssetType.IMAGE, AssetType.MARKDOWN_BLOCK],
    )
    assert not embedding_type_supports_any_asset_type(
        embedding_type=EmbeddingType.VISUAL_PRESENTATION,
        asset_types=[AssetType.MARKDOWN_BLOCK, AssetType.TEXT_BLOCK],
    )
    assert embedding_type_supports_any_asset_type(
        embedding_type=EmbeddingType.SUBJECT_CONTENT,
        asset_types=[AssetType.MARKDOWN_BLOCK, AssetType.TEXT_BLOCK],
    )


def test_null_feature_statuses_never_expose_stale_text() -> None:
    for status in ("unknown", "not_applicable"):
        features = {
            "scene_theme": {
                "value": "错误遗留的场景主题",
                "status": status,
            }
        }

        assert (
            effective_feature_text(
                features,
                EmbeddingType.SCENE_THEME,
            )
            is None
        )
        assert not embedding_channel_is_eligible(
            embedding_type=EmbeddingType.SCENE_THEME,
            asset_features=features,
        )


def test_explicit_null_effective_value_does_not_fall_back_to_model_value() -> None:
    features = {
        "visual_presentation": {
            "effective_value": None,
            "model_value": "赛博朋克",
            "status": "observed",
        }
    }

    assert effective_feature_text(features, EmbeddingType.VISUAL_PRESENTATION) is None


def test_current_and_spec_feature_shapes_are_both_supported() -> None:
    current = {"visual_presentation": {"value": "赛博朋克", "status": "observed"}}
    specified = {
        "visual_presentation": {
            "model_value": "写实电影感",
            "user_value": "水彩",
            "status": "user_supplied",
        }
    }

    assert effective_feature_text(current, EmbeddingType.VISUAL_PRESENTATION) == "赛博朋克"
    assert effective_feature_text(specified, EmbeddingType.VISUAL_PRESENTATION) == "水彩"


def test_visual_presentation_reads_legacy_style_and_color_features() -> None:
    features = {
        "visual_style": {"value": "写实数字渲染", "status": "observed"},
        "color_composition": {"value": "冷蓝色；中心构图", "status": "observed"},
    }

    assert effective_feature_text(features, EmbeddingType.VISUAL_PRESENTATION) == (
        "写实数字渲染；冷蓝色；中心构图"
    )


def test_salience_items_are_serialized_into_one_dimension_text() -> None:
    features = {
        "subject_content": {
            "applicability": "applicable",
            "items": [
                {
                    "subject": "远山",
                    "description": "背景中的远山",
                    "salience": 0.2,
                    "status": "observed",
                    "evidence": ["画面远处可见山体"],
                },
                {
                    "subject": "女孩",
                    "description": "手持雨伞的女孩",
                    "salience": 0.9,
                    "status": "observed",
                    "evidence": ["女孩位于画面中央"],
                },
                {
                    "subject": "小狗",
                    "description": "跟随女孩的小狗",
                    "salience": 0.5,
                    "status": "observed",
                    "evidence": ["小狗位于女孩身后"],
                },
            ],
        }
    }

    assert effective_feature_text(features, EmbeddingType.SUBJECT_CONTENT) == (
        "核心信息：女孩：手持雨伞的女孩；重要补充：小狗：跟随女孩的小狗；"
        "辅助信息：远山：背景中的远山"
    )
