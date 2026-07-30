import json

from capsule.enums import ClusterInternalVariance, ClusterRepresentativeRole, EmbeddingType
from capsule.pipeline.cluster_summary import (
    CLUSTER_SUMMARY_DIMENSION_POLICIES,
    ClusterSummaryRepresentative,
    asset_usage_path_context,
    build_cluster_summary_messages,
    cluster_source_context,
    ensure_asset_usage_cluster_path_description,
    ensure_path_aware_cluster_summary,
)
from capsule.schemas import ClusterSummary


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
    assert "同一概念只保留一个最准确" in prompt


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


def test_asset_usage_summary_receives_and_guarantees_relative_path_context() -> None:
    source_paths = [
        "海报/素材/20251216-143446.png",
        "海报/素材/20251216-143450.png",
        "海报/png/111.png",
    ]
    representative = ClusterSummaryRepresentative(
        asset_id="asset_usage",
        role=ClusterRepresentativeRole.MEDOID,
        asset_type="image",
        asset_name="海报视觉",
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
        file_tree_context=["海报", "素材"],
        membership_probability=1.0,
        distance_to_medoid=0.0,
        source_relative_path="海报/素材/20251216-143446.png",
    )

    messages = build_cluster_summary_messages(
        embedding_type="asset_usage",
        member_count=3,
        average_membership_probability=0.95,
        representatives=[representative],
        member_source_paths=source_paths,
    )
    payload = json.loads(messages[1]["content"])
    path_context = payload["member_source_context"]
    evidence = payload["representative_assets"][0]

    assert path_context == asset_usage_path_context(source_paths)
    assert path_context["directory_counts"][0] == {
        "directory": "海报/素材",
        "member_count": 2,
    }
    assert evidence["source_relative_path"] == "海报/素材/20251216-143446.png"
    assert evidence["source_file_name"] == "20251216-143446.png"
    assert "必须明确写出" in messages[0]["content"]

    description = ensure_asset_usage_cluster_path_description(
        "本组资产主要用于宣传海报的视觉设计和制作，可服务于推广物料的统一交付。",
        source_paths,
    )
    assert "2项来自「海报/素材」" in description
    assert "海报/素材/20251216-143446.png" in description
    assert len(description) <= 150


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
    representative = ClusterSummaryRepresentative(
        asset_id="asset_guxiaoling",
        role=ClusterRepresentativeRole.MEDOID,
        asset_type="image",
        asset_name="双色马尾动漫女孩古小玲立绘",
        asset_description="一名二次元少女角色的立绘。",
        asset_features={"subject_content": {"value": "二次元少女角色立绘"}},
        file_tree_context=["第一集", "古小玲", "古小玲"],
        membership_probability=0.98,
        distance_to_medoid=0.0,
        source_relative_path=source_paths[0],
    )

    messages = build_cluster_summary_messages(
        embedding_type="subject_content",
        member_count=len(source_paths),
        average_membership_probability=0.93,
        representatives=[representative],
        member_source_paths=source_paths,
    )
    payload = json.loads(messages[1]["content"])
    context = payload["member_source_context"]
    evidence = payload["representative_assets"][0]

    assert context["semantic_path_terms"][0] == {
        "term": "古小玲",
        "member_count": 2,
    }
    assert evidence["source_relative_path"] == source_paths[0]
    assert evidence["source_file_name"] == "2√.png"
    assert "项目名或角色名" in messages[0]["content"]

    summary = ClusterSummary(
        name="二次元少女立绘",
        description=(
            "本组资产的核心主体均为二次元动漫少女，多呈现人物立绘内容，在服饰、发型和"
            "姿态等具体表现上存在一定差异。"
        ),
        keywords=["二次元", "少女", "立绘"],
        common_features=["少女角色"],
        internal_variance=ClusterInternalVariance.MEDIUM,
    )
    enriched = ensure_path_aware_cluster_summary(
        summary,
        source_paths,
        embedding_type="subject_content",
    )

    assert enriched.name == "古小玲及其他二次元少女立绘"
    assert "「古小玲」出现在2项成员中" in enriched.description
    assert "代表文件「2√.png」" in enriched.description
    assert source_paths[0] in enriched.description
    assert len(enriched.description) <= 150


def test_path_entity_does_not_duplicate_an_existing_subject_name() -> None:
    source_paths = [
        "第一集/古小玲/1208-三视图/汪叹之/4.png",
        "第一集/古小玲/1208-三视图/汪叹之/正视图√.png",
        "第一集/汪叹之/1/23333332.png",
        "第一集/汪叹之/20251127-121853.png",
    ]
    summary = ClusterSummary(
        name="男性汪叹之立绘",
        description=(
            "本组资产的核心主体均为男性角色汪叹之，呈现形式均为人物立绘，局部服饰和"
            "视角存在轻微差异，其中既有正面视图，也包含背面和半身视图等补充内容。"
        ),
        keywords=["汪叹之", "男性", "立绘"],
        common_features=["男性角色"],
        internal_variance=ClusterInternalVariance.LOW,
    )

    enriched = ensure_path_aware_cluster_summary(
        summary,
        source_paths,
        embedding_type="subject_content",
    )

    assert enriched.name == "男性汪叹之立绘"
    assert "「汪叹之」出现在4项成员中" in enriched.description


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


def test_long_file_path_is_compacted_without_cutting_the_description_mid_path() -> None:
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
        description=(
            "本组资产主要用于动画角色设计参考，成员具有相似的人物呈现方式，可为角色造型"
            "和视觉设定环节提供素材依据"
        ),
        keywords=["动画", "角色", "参考"],
        common_features=["角色设计"],
        internal_variance=ClusterInternalVariance.LOW,
    )

    enriched = ensure_path_aware_cluster_summary(
        summary,
        source_paths,
        embedding_type="asset_usage",
    )

    assert "相对目录「第一集/小说编辑/测试」" in enriched.description
    assert "代表文件「" in enriched.description
    assert enriched.description.endswith("。")
    assert len(enriched.description) <= 150
