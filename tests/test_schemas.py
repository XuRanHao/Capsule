import math

import pytest
from pydantic import ValidationError

from capsule.enums import FeatureStatus
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
                "embedding_type": "visual_style",
                "confidence": "not-a-number",
                "evidence": None,
            },
            {
                "key": "color_composition",
                "name": "color_composition",
                "value": "蓝紫色",
            },
        ]
    )

    assert understanding.features.subject_content.confidence == 0.75
    assert understanding.features.subject_content.evidence == ["画面中央可见圆形角色"]
    assert understanding.features.scene_theme.status is FeatureStatus.INFERRED
    assert understanding.features.scene_theme.confidence == 1.0
    assert understanding.features.scene_theme.evidence == ["背景为夜间城市"]
    assert understanding.features.visual_style.value is None
    assert understanding.features.visual_style.status is FeatureStatus.UNKNOWN
    assert understanding.features.visual_style.confidence == 0.0
    assert understanding.features.color_composition.status is FeatureStatus.INFERRED
    assert understanding.features.target_audience.status is FeatureStatus.UNKNOWN


@pytest.mark.parametrize(
    ("raw_confidence", "expected"),
    [
        ("0.25", 0.25),
        ("-0.25", 0.0),
        ("1.25", 1.0),
        ("NaN", 0.0),
        (float("inf"), 0.0),
        (float("-inf"), 0.0),
        ("invalid", 0.0),
        (None, 0.0),
        (True, 0.0),
    ],
)
def test_feature_value_normalizes_confidence(
    raw_confidence: object,
    expected: float,
) -> None:
    feature = FeatureValue.model_validate(
        {
            "value": "测试值",
            "status": "observed",
            "confidence": raw_confidence,
        }
    )

    assert math.isfinite(feature.confidence)
    assert feature.confidence == expected
    assert feature.evidence == []


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
