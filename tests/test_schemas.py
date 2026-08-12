import pytest
from pydantic import ValidationError

from capsule.enums import FeatureApplicability, FeatureSalience, FeatureStatus
from capsule.schemas import AssetUnderstanding, FeatureValue


def _understanding(features: object) -> AssetUnderstanding:
    return AssetUnderstanding.model_validate(
        {
            "asset_name": "测试素材",
            "asset_description": "用于验证模型输出归一化的测试素材。",
            "features": features,
        }
    )


def test_asset_understanding_normalizes_unambiguous_feature_array() -> None:
    understanding = _understanding(
        [
            {
                "key": "subject_content",
                "value": "圆形角色",
                "status": "observed",
                "confidence": "0.75",
                "evidence": "画面中央可见圆形角色",
            },
            {
                "name": "scene_theme",
                "value": "都市夜景",
                "confidence": "1.5",
                "evidence": ["背景为夜间城市", "第二条证据会被现有规则截断"],
            },
            {
                "key": "visual_presentation",
                "name": "visual_presentation",
                "value": "蓝紫色",
            },
        ]
    )

    subject = understanding.features.subject_content
    assert subject.applicability is FeatureApplicability.APPLICABLE
    assert subject.items[0].subject == "圆形角色"
    assert subject.items[0].description == "圆形角色"
    assert subject.items[0].status is FeatureStatus.OBSERVED
    assert subject.items[0].evidence == ["画面中央可见圆形角色"]
    scene = understanding.features.scene_theme
    assert scene.items[0].status is FeatureStatus.INFERRED
    assert scene.items[0].evidence == ["背景为夜间城市"]
    assert (
        understanding.features.visual_presentation.items[0].status
        is FeatureStatus.INFERRED
    )


def test_subject_content_preserves_subject_separately_from_description() -> None:
    understanding = _understanding(
        {
            "subject_content": {
                "applicability": "applicable",
                "items": [
                    {
                        "subject": "枉叹之",
                        "description": "黑发男性角色，手持长剑并面向前方站立",
                        "salience": "high",
                        "status": "metadata",
                        "evidence": ["文件名与画面中的角色一致"],
                        "ocr_confidence": None,
                    }
                ],
            }
        }
    )

    item = understanding.features.subject_content.items[0]
    assert item.subject == "枉叹之"
    assert item.description == "黑发男性角色，手持长剑并面向前方站立"


def test_subject_content_sorts_by_relative_salience() -> None:
    understanding = _understanding(
        {
            "subject_content": {
                "applicability": "applicable",
                "items": [
                    {
                        "subject": "封不觉",
                        "description": "黑发红眼男性角色",
                        "salience": 0.92,
                    },
                    {
                        "subject": "活动管钳",
                        "description": "角色手持的红色管钳",
                        "salience": 0.48,
                    },
                ],
            }
        }
    )

    items = understanding.features.subject_content.items
    assert [item.salience for item in items] == [0.92, 0.48]


def test_asset_understanding_merges_legacy_visual_fields() -> None:
    understanding = _understanding(
        {
            "visual_style": {"value": "数字插画；写实质感", "status": "observed"},
            "color_composition": {"value": "冷蓝色；中心构图", "status": "observed"},
        }
    )

    assert understanding.features.visual_presentation.value == (
        "数字插画；冷蓝色；写实质感；中心构图"
    )


def test_feature_value_normalizes_salience_items_and_drops_feature_confidence() -> None:
    feature = FeatureValue.model_validate(
        {
            "applicability": "applicable",
            "items": [
                {
                    "description": "背景车辆",
                    "salience": "low",
                    "status": "observed",
                    "evidence": "画面后方可见车辆",
                    "ocr_confidence": "invalid",
                },
                {
                    "description": "中央角色",
                    "salience": "high",
                    "status": "observed",
                    "evidence": ["角色位于中央", "多余证据"],
                    "ocr_confidence": 0.92,
                },
            ],
        }
    )

    assert [item.description for item in feature.items] == ["中央角色", "背景车辆"]
    assert feature.items[0].salience is FeatureSalience.HIGH
    assert feature.items[0].evidence == ["角色位于中央"]
    assert feature.items[0].ocr_confidence == 0.92
    assert feature.items[1].ocr_confidence is None
    dumped = feature.model_dump(mode="json")
    assert "confidence" not in dumped
    assert "value" not in dumped


@pytest.mark.parametrize(
    "features",
    [
        [],
        ["subject_content"],
        [{"feature_name": "subject_content", "value": "角色"}],
        [{"key": "native_multimodal", "value": "角色"}],
        [
            {
                "key": "subject_content",
                "name": "scene_theme",
                "value": "角色",
            }
        ],
        [
            {"key": "subject_content", "value": "角色"},
            {"name": "subject_content", "value": "另一个角色"},
        ],
    ],
)
def test_asset_understanding_rejects_ambiguous_feature_arrays(
    features: object,
) -> None:
    with pytest.raises(ValidationError, match="features must be an object"):
        _understanding(features)
