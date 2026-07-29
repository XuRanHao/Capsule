import json

from capsule.enums import ClusterRepresentativeRole
from capsule.pipeline.cluster_summary import (
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
