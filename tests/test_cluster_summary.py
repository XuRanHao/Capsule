import json

from capsule.enums import ClusterInternalVariance, EmbeddingType
from capsule.pipeline.cluster_summary import (
    CLUSTER_SUMMARY_DIMENSION_POLICIES,
    ClusterSummaryAsset,
    asset_usage_path_context,
    build_cluster_summary_messages,
    cluster_source_context,
    ensure_path_aware_cluster_summary,
)
from capsule.schemas import ClusterSummary


def test_cluster_summary_input_contains_every_asset_description_and_dimension() -> None:
    messages = build_cluster_summary_messages(
        embedding_type="visual_style",
        member_count=42,
        average_membership_probability=0.81,
        assets=[
            ClusterSummaryAsset(
                asset_id="asset_medoid",
                asset_description="蓝紫色霓虹灯下的夜间街道与人物。",
                asset_features={"visual_style": {"effective_value": "赛博朋克"}},
            ),
            ClusterSummaryAsset(
                asset_id="asset_edge",
                asset_description="夜景镜头缓慢推进，霓虹灯反射在湿润路面。",
                asset_features={"visual_style": {"value": "写实电影感"}},
            ),
        ],
    )

    payload = json.loads(messages[1]["content"])

    assert payload["semantic_dimension"] == "视觉风格"
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
        embedding_type.value for embedding_type in EmbeddingType
    }


def test_color_composition_summary_is_restricted_to_its_feature_dimension() -> None:
    messages = build_cluster_summary_messages(
        embedding_type="color_composition",
        member_count=6,
        average_membership_probability=0.97,
        assets=[
            ClusterSummaryAsset(
                asset_id="asset_color",
                asset_description="一张以动漫人物为主体的宣传海报。",
                asset_features={
                    "subject_content": {"value": "动漫人物；宣传主题"},
                    "color_composition": {"value": "暗调；低饱和度；暖色点缀；强明暗对比"},
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
        "current_dimension_description": "暗调；低饱和度；暖色点缀；强明暗对比",
    }
    assert policy["title_focus"].startswith("最有区分度的色彩关系")
    assert "视角、景别、画面布局、空间层次和视觉重心" in policy["description_focus"]
    assert "description_must_exclude" not in policy
    assert "title_must_exclude" not in policy
    assert "description_must_exclude" not in prompt
    assert "title_must_exclude" not in prompt
    assert "尽可能准确、完整且简洁" in prompt
    assert "共同定义当前任务的正向语义范围" in prompt
    assert "成员差异通过 internal_variance 表达" in prompt
    assert "common_features 必须有 1 到 3 项" in prompt
    assert "30 到 80" in prompt
    assert "不要输出 keywords" in prompt
    assert "说明该维度内的必要差异" not in prompt


def test_description_channels_only_receive_their_permitted_text_evidence() -> None:
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
    description_payload = json.loads(
        build_cluster_summary_messages(
            embedding_type="asset_description",
            member_count=1,
            average_membership_probability=1.0,
            assets=[cluster_asset],
        )[1]["content"]
    )

    native_evidence = native_payload["cluster_assets"][0]
    description_evidence = description_payload["cluster_assets"][0]
    assert native_evidence["asset_description"] == "蓝色海面上有一艘白色帆船。"
    assert native_evidence["current_dimension_description"] == "蓝色海面上有一艘白色帆船。"
    assert description_evidence["asset_description"] == "蓝色海面上有一艘白色帆船。"
    assert description_evidence["current_dimension_description"] == (
        "蓝色海面上有一艘白色帆船。"
    )


def test_asset_usage_summary_keeps_path_context_as_evidence_metadata() -> None:
    source_paths = [
        "海报/素材/20251216-143446.png",
        "海报/素材/20251216-143450.png",
        "海报/png/111.png",
    ]
    cluster_asset = ClusterSummaryAsset(
        asset_id="asset_usage",
        asset_description="一张宣传海报。",
        asset_features={
            "asset_usage": {
                "value": "海报制作",
                "status": "metadata",
                "description": (
                    "该素材对应相对文件路径「海报/素材/20251216-143446.png」，用于海报制作。"
                ),
                "source_path": "海报/素材/20251216-143446.png",
            }
        },
        source_relative_path="海报/素材/20251216-143446.png",
    )

    messages = build_cluster_summary_messages(
        embedding_type="asset_usage",
        member_count=3,
        average_membership_probability=0.95,
        assets=[cluster_asset],
        member_source_paths=source_paths,
    )
    payload = json.loads(messages[1]["content"])
    path_context = payload["member_source_context"]
    evidence = payload["cluster_assets"][0]

    assert path_context == asset_usage_path_context(source_paths)
    assert path_context["directory_counts"][0] == {
        "directory": "海报/素材",
        "member_count": 2,
    }
    assert evidence["source_relative_path"] == "海报/素材/20251216-143446.png"
    assert evidence["source_file_name"] == "20251216-143446.png"
    assert "成员数量、完整路径、文件名和目录统计作为证据元数据保留" in messages[0]["content"]


def test_subject_content_summary_preserves_named_path_entity_and_file_evidence() -> None:
    source_paths = [
        "第一集/古小玲/古小玲/2√.png",
        "第一集/古小玲/古小玲/4√.png",
        "第一集/小说编辑/参考/1ea70fc6ff5855158758b2aea45b4519.jpg",
        "第一集/小说编辑/参考/20251229-135033.jpg",
        "第一集/小说编辑/参考/333bc72f4295bdba2f37068cbf4893ad.jpg",
        "第一集/小说编辑/参考/6421538afae524f1035ff2460e20a835.jpg",
        "第一集/小说编辑/参考/bb1f3e2600aec859f35e971d107330ac.jpg",
        "第一集/小说编辑/参考/f9951fe08005132c82c7b6c57fcd6cd3.jpg",
    ]
    cluster_asset = ClusterSummaryAsset(
        asset_id="asset_guxiaoling",
        asset_description="一名二次元少女角色的立绘。",
        asset_features={"subject_content": {"value": "二次元少女角色立绘"}},
        source_relative_path=source_paths[0],
    )

    messages = build_cluster_summary_messages(
        embedding_type="subject_content",
        member_count=len(source_paths),
        average_membership_probability=0.93,
        assets=[cluster_asset],
        member_source_paths=source_paths,
    )
    payload = json.loads(messages[1]["content"])
    context = payload["member_source_context"]
    evidence = payload["cluster_assets"][0]

    assert context["semantic_path_terms"][0] == {
        "term": "古小玲",
        "member_count": 2,
    }
    assert evidence["source_relative_path"] == source_paths[0]
    assert evidence["source_file_name"] == "2√.png"
    assert "代表性语义实体" in messages[0]["content"]

    summary = ClusterSummary(
        name="二次元少女立绘",
        description="共同呈现二次元少女角色立绘，人物主体和立绘形式在代表资产中保持一致。",
        common_features=["少女角色"],
        internal_variance=ClusterInternalVariance.MEDIUM,
    )
    enriched = ensure_path_aware_cluster_summary(
        summary,
        source_paths,
        embedding_type="subject_content",
    )

    assert enriched.name == "古小玲及其他二次元少女立绘"
    assert enriched.description == summary.description
    assert source_paths[0] not in enriched.description


def test_path_entity_does_not_duplicate_an_existing_subject_name() -> None:
    source_paths = [
        "第一集/古小玲/1208-三视图/汪叹之/4.png",
        "第一集/古小玲/1208-三视图/汪叹之/正视图√.png",
        "第一集/汪叹之/1/23333332.png",
        "第一集/汪叹之/20251127-121853.png",
    ]
    summary = ClusterSummary(
        name="男性汪叹之立绘",
        description="共同呈现男性角色汪叹之立绘，人物身份与立绘形式在代表资产中保持稳定一致。",
        common_features=["男性角色"],
        internal_variance=ClusterInternalVariance.LOW,
    )

    enriched = ensure_path_aware_cluster_summary(
        summary,
        source_paths,
        embedding_type="subject_content",
    )

    assert enriched.name == "男性汪叹之立绘"
    assert enriched.description == summary.description


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


def test_long_file_path_is_not_injected_into_cluster_description() -> None:
    long_name = (
        "MoriMai_httpss.mj.runPIXbomaA8kc_An_anime-style_digital_illus_"
        "59b7f9ba-6677-40d3-8479-acc48a9b26ff_3.png"
    )
    source_paths = [
        f"第一集/小说编辑/测试/{long_name}",
        f"第一集/小说编辑/测试/copy_{long_name}",
    ]
    summary = ClusterSummary(
        name="动画角色参考",
        description="共同用于动画角色设计参考，角色造型和视觉设定用途在代表资产中保持一致。",
        common_features=["角色设计"],
        internal_variance=ClusterInternalVariance.LOW,
    )

    enriched = ensure_path_aware_cluster_summary(
        summary,
        source_paths,
        embedding_type="asset_usage",
    )

    assert enriched.description == summary.description
    assert long_name not in enriched.description
