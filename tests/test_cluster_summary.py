import json

from capsule.enums import ClusterRepresentativeRole, EmbeddingType
from capsule.pipeline.cluster_summary import (
    CLUSTER_SUMMARY_DIMENSION_POLICIES,
    ClusterSummaryRepresentative,
    build_cluster_summary_messages,
)


def test_cluster_summary_input_contains_only_selected_representative_assets() -> None:
    messages = build_cluster_summary_messages(
        embedding_type="visual_style",
        member_count=42,
        average_membership_probability=0.81,
        representatives=[
            ClusterSummaryRepresentative(
                asset_id="asset_medoid",
                role=ClusterRepresentativeRole.MEDOID,
                asset_type="image",
                asset_name="霓虹街道",
                asset_description="蓝紫色霓虹灯下的夜间街道与人物。",
                asset_features={"visual_style": {"effective_value": "赛博朋克"}},
                file_tree_context=["inspiration", "night"],
                membership_probability=0.98,
                distance_to_medoid=0.0,
            ),
            ClusterSummaryRepresentative(
                asset_id="asset_edge",
                role=ClusterRepresentativeRole.EDGE,
                asset_type="video_segment",
                asset_name=None,
                asset_description="夜景镜头缓慢推进，霓虹灯反射在湿润路面。",
                asset_features={"visual_style": {"value": "写实电影感"}},
                file_tree_context=["inspiration", "video"],
                membership_probability=0.3,
                distance_to_medoid=1.2,
            ),
        ],
    )

    payload = json.loads(messages[1]["content"])

    assert payload["semantic_dimension"] == "视觉风格"
    assert [item["asset_id"] for item in payload["representative_assets"]] == [
        "asset_medoid",
        "asset_edge",
    ]
    assert "asset_discarded" not in messages[1]["content"]
    assert payload["representative_assets"][0]["current_dimension_feature"] == "赛博朋克"
    assert payload["representative_assets"][1]["current_dimension_feature"] == "写实电影感"
    assert "asset_name" not in payload["representative_assets"][0]
    assert "asset_description" not in payload["representative_assets"][0]
    assert "asset_type" not in payload["representative_assets"][0]
    assert "file_tree_context" not in payload["representative_assets"][0]


def test_cluster_summary_policies_cover_every_embedding_type() -> None:
    assert set(CLUSTER_SUMMARY_DIMENSION_POLICIES) == {
        embedding_type.value for embedding_type in EmbeddingType
    }


def test_color_composition_summary_is_restricted_to_its_feature_dimension() -> None:
    messages = build_cluster_summary_messages(
        embedding_type="color_composition",
        member_count=6,
        average_membership_probability=0.97,
        representatives=[
            ClusterSummaryRepresentative(
                asset_id="asset_color",
                role=ClusterRepresentativeRole.MEDOID,
                asset_type="image",
                asset_name="暗黑动漫宣传海报",
                asset_description="一张以动漫人物为主体的宣传海报。",
                asset_features={
                    "subject_content": {"value": "动漫人物；宣传主题"},
                    "color_composition": {"value": "暗调；低饱和度；暖色点缀；强明暗对比"},
                },
                file_tree_context=["海报", "动漫"],
                membership_probability=0.99,
                distance_to_medoid=0.0,
            )
        ],
    )

    prompt = messages[0]["content"]
    payload = json.loads(messages[1]["content"])
    representative = payload["representative_assets"][0]
    policy = payload["dimension_policy"]

    assert representative == {
        "asset_id": "asset_color",
        "role": "medoid",
        "membership_probability": 0.99,
        "distance_to_medoid": 0.0,
        "current_dimension_feature": "暗调；低饱和度；暖色点缀；强明暗对比",
    }
    assert policy["title_focus"].startswith("只提炼颜色")
    assert {"海报", "插画", "动漫", "主题", "场景", "宣传"} <= set(policy["title_must_exclude"])
    assert prompt.index('"description"') < prompt.index('"name"')
    assert "先写 description" in prompt
    assert "再从已经写好的 description" in prompt


def test_description_channels_only_receive_their_permitted_text_evidence() -> None:
    representative = ClusterSummaryRepresentative(
        asset_id="asset_text",
        role=ClusterRepresentativeRole.MEDOID,
        asset_type="image",
        asset_name="文件名称",
        asset_description="蓝色海面上有一艘白色帆船。",
        asset_features={"subject_content": {"value": "帆船"}},
        file_tree_context=["旅行", "参考"],
        membership_probability=1.0,
        distance_to_medoid=0.0,
    )

    native_payload = json.loads(
        build_cluster_summary_messages(
            embedding_type="native_multimodal",
            member_count=1,
            average_membership_probability=1.0,
            representatives=[representative],
        )[1]["content"]
    )
    description_payload = json.loads(
        build_cluster_summary_messages(
            embedding_type="asset_description",
            member_count=1,
            average_membership_probability=1.0,
            representatives=[representative],
        )[1]["content"]
    )

    native_evidence = native_payload["representative_assets"][0]
    description_evidence = description_payload["representative_assets"][0]
    assert native_evidence["asset_name"] == "文件名称"
    assert native_evidence["asset_description"] == "蓝色海面上有一艘白色帆船。"
    assert description_evidence["asset_description"] == "蓝色海面上有一艘白色帆船。"
    assert "asset_name" not in description_evidence
    assert "file_tree_context" not in native_evidence
    assert "current_dimension_feature" not in description_evidence
