import json

from capsule.features import ACTIVE_EMBEDDING_TYPES
from capsule.pipeline.cluster_summary import (
    CLUSTER_SUMMARY_DIMENSION_POLICIES,
    ClusterSummaryAsset,
    build_cluster_summary_messages,
    cluster_source_context,
)


def test_cluster_summary_input_contains_every_asset_description_and_dimension() -> None:
    messages = build_cluster_summary_messages(
        embedding_type="visual_presentation",
        member_count=42,
        average_membership_probability=0.81,
        assets=[
            ClusterSummaryAsset(
                asset_id="asset_medoid",
                asset_description="蓝紫色霓虹灯下的夜间街道与人物。",
                asset_features={"visual_presentation": {"effective_value": "赛博朋克"}},
            ),
            ClusterSummaryAsset(
                asset_id="asset_edge",
                asset_description="夜景镜头缓慢推进，霓虹灯反射在湿润路面。",
                asset_features={"visual_presentation": {"value": "写实电影感"}},
            ),
        ],
    )

    payload = json.loads(messages[1]["content"])

    assert payload["semantic_dimension"] == "视觉表现"
    assert [item["asset_id"] for item in payload["cluster_assets"]] == [
        "asset_medoid",
        "asset_edge",
    ]
    assert "asset_discarded" not in messages[1]["content"]
    assert payload["cluster_assets"][0] == {
        "asset_id": "asset_medoid",
        "asset_description": "蓝紫色霓虹灯下的夜间街道与人物。",
        "current_dimension_description": "赛博朋克",
    }
    assert payload["cluster_assets"][1]["asset_description"] == (
        "夜景镜头缓慢推进，霓虹灯反射在湿润路面。"
    )
    assert payload["cluster_assets"][1]["current_dimension_description"] == "写实电影感"
    assert "representative_assets" not in payload


def test_cluster_summary_policies_cover_every_embedding_type() -> None:
    assert set(CLUSTER_SUMMARY_DIMENSION_POLICIES) == {
        embedding_type.value for embedding_type in ACTIVE_EMBEDDING_TYPES
    }


def test_visual_presentation_summary_combines_style_color_and_composition() -> None:
    messages = build_cluster_summary_messages(
        embedding_type="visual_presentation",
        member_count=6,
        average_membership_probability=0.97,
        assets=[
            ClusterSummaryAsset(
                asset_id="asset_color",
                asset_description="一张以动漫人物为主体的宣传海报。",
                asset_features={
                    "subject_content": {"value": "动漫人物；宣传主题"},
                    "visual_presentation": {
                        "value": "写实插画；暗调；低饱和度；暖色点缀；强明暗对比"
                    },
                },
            )
        ],
    )

    prompt = messages[0]["content"]
    payload = json.loads(messages[1]["content"])
    cluster_asset = payload["cluster_assets"][0]
    policy = payload["dimension_policy"]

    assert cluster_asset == {
        "asset_id": "asset_color",
        "asset_description": "一张以动漫人物为主体的宣传海报。",
        "current_dimension_description": "写实插画；暗调；低饱和度；暖色点缀；强明暗对比",
    }
    assert set(policy) == {"summary_focus"}
    assert "视角、景别、透视、留白、均衡、节奏" in policy["summary_focus"]
    assert "跨主体、跨场景找到呈现方式相近" in policy["summary_focus"]
    assert "description_must_exclude" not in prompt
    assert "title_must_exclude" not in prompt
    assert "同时生成彼此一致的 name 和 description" in prompt
    assert "完全回退到原有按聚类维度总结的路径" in prompt
    assert "成员差异通过 internal_variance 表达" in prompt
    assert "1 到 3 项事实" in prompt
    assert "30 到 80" in prompt
    assert "不要输出 keywords" in prompt
    assert "说明该维度内的必要差异" not in prompt


def test_native_channel_receives_asset_description_as_text_evidence() -> None:
    cluster_asset = ClusterSummaryAsset(
        asset_id="asset_text",
        asset_description="蓝色海面上有一艘白色帆船。",
        asset_features={"subject_content": {"value": "帆船"}},
    )

    native_payload = json.loads(
        build_cluster_summary_messages(
            embedding_type="native_multimodal",
            member_count=1,
            average_membership_probability=1.0,
            assets=[cluster_asset],
        )[1]["content"]
    )
    native_evidence = native_payload["cluster_assets"][0]
    assert native_evidence["asset_description"] == "蓝色海面上有一艘白色帆船。"
    assert native_evidence["current_dimension_description"] == "蓝色海面上有一艘白色帆船。"


def test_cluster_summary_uses_effective_metadata_to_infer_user_intent() -> None:
    source_paths = [
        "角色资产/风不觉-武器.png",
        "角色资产/风不觉-角色设定.png",
        "角色资产/风不觉-时装.png",
    ]
    assets = [
        ClusterSummaryAsset(
            asset_id=f"asset_{index}",
            asset_description=description,
            asset_features={"visual_presentation": {"value": "二次元角色设定图"}},
            source_relative_path=source_path,
            file_tree_context=("角色资产",),
        )
        for index, (source_path, description) in enumerate(
            zip(
                source_paths,
                ["角色使用的武器。", "角色造型设定。", "角色服装设计。"],
                strict=True,
            )
        )
    ]

    messages = build_cluster_summary_messages(
        embedding_type="visual_presentation",
        member_count=len(source_paths),
        average_membership_probability=0.93,
        assets=assets,
        member_source_paths=source_paths,
    )
    payload = json.loads(messages[1]["content"])
    context = payload["member_metadata_context"]
    evidence = payload["cluster_assets"][0]

    assert evidence["source_relative_path"] == source_paths[0]
    assert evidence["source_file_name"] == "风不觉-武器.png"
    assert evidence["file_tree_context"] == ["角色资产"]
    assert context["member_count_with_path"] == 3
    prompt = messages[0]["content"]
    assert "用户在组织“风不觉”相关资产" in prompt
    assert "不得机械截取共同字符串" in prompt
    assert "name 和 description 都必须作为同一次判断的整体直接生成" in prompt


def test_generic_path_folders_do_not_become_title_entities() -> None:
    context = cluster_source_context(
        [
            "第一集/小说编辑/测试/preview.png",
            "第一集/小说编辑/测试/final.png",
            "海报/png/海报1.png",
            "海报/png/海报2.png",
        ]
    )

    assert context["semantic_path_terms"] == []


def test_uninformative_metadata_falls_back_to_dimension_summary() -> None:
    long_name = (
        "MoriMai_httpss.mj.runPIXbomaA8kc_An_anime-style_digital_illus_"
        "59b7f9ba-6677-40d3-8479-acc48a9b26ff_3.png"
    )
    source_paths = [
        f"第一集/小说编辑/测试/{long_name}",
        f"第一集/小说编辑/测试/copy_{long_name}",
    ]
    messages = build_cluster_summary_messages(
        embedding_type="subject_content",
        member_count=2,
        average_membership_probability=0.9,
        assets=[
            ClusterSummaryAsset(
                asset_id=f"asset_{index}",
                asset_description="动画角色设计参考。",
                asset_features={"subject_content": {"value": "动画角色"}},
                source_relative_path=source_path,
            )
            for index, source_path in enumerate(source_paths)
        ],
        member_source_paths=source_paths,
    )

    prompt = messages[0]["content"]
    assert "UUID、通用目录、临时/导出标记和孤立的单个命名都不是有效意图证据" in prompt
    assert "完全回退到原有按聚类维度总结的路径" in prompt
