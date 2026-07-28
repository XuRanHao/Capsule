"use client";

import { FormEvent, KeyboardEvent, useMemo, useState } from "react";
import { ProductTopbar } from "./components/DemoShell";

type QueryType = "text" | "image" | "image_text";
type AssetType = "image" | "video_segment" | "markdown_block";
type FusionMethod = "weighted_rrf" | "normalized_weighted_similarity";

type DimensionQuery = {
  embedding_type: string;
  query: string;
  weight: number;
  source: "text" | "image" | "joint";
  constraint: string;
};

type ParsedQuery = {
  query_summary: string;
  dimension_queries: DimensionQuery[];
  negative_terms: string[];
  parser_mode: string;
};

type MatchedChannel = {
  channel: string;
  embedding_type: string;
  rank: number;
  similarity: number;
  fusion_contribution: number;
  rrf_contribution: number;
};

type SearchResult = {
  asset_id: string;
  asset_type: AssetType;
  asset_name: string | null;
  asset_description: string | null;
  asset_features: Record<string, unknown>;
  source_contexts: Array<{
    text?: string;
    relation_type?: string;
    text_block_index?: number;
  }>;
  source_locator: Record<string, unknown>;
  preview_uri: string | null;
  source_file: {
    source_file_id: string;
    original_file_name: string;
    file_type: string;
    relative_path: string;
  } | null;
  score: number;
  matched_channels: MatchedChannel[];
  matched_feature: string | null;
  matched_reason: string | null;
  rerank_score: number | null;
  group_kind: string | null;
  folded_asset_ids: string[];
  available: boolean;
};

type SearchResponse = {
  query: {
    query_type: QueryType;
    query_text: string | null;
    query_image_url: string | null;
    query_image_upload_id: string | null;
    precision_mode: boolean;
  };
  parsed_query: ParsedQuery | null;
  fusion_method: FusionMethod;
  rerank_method: "off" | "doubao_seed_2_lite";
  search_engine_version: string;
  execution_id: string | null;
  capsule_id: string | null;
  total: number;
  degraded: boolean;
  degraded_reasons: string[];
  timings: { total_ms: number };
  results: SearchResult[];
};

type CapsuleSummary = {
  capsule_id: string;
  query_type: QueryType;
  query_text: string | null;
  query_image_uri: string | null;
  query_summary: string;
  fusion_method: FusionMethod;
  rerank_method: string;
  is_favorite: boolean;
  result_count: number;
  last_used_at: string;
  created_at: string;
};

type CapsuleDetail = CapsuleSummary & {
  parsed_query: ParsedQuery;
  latest_snapshot: {
    execution_id: string;
    created_at: string;
    results: SearchResult[];
  };
  executions: string[];
};

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8010";

const QUERY_TYPES: Array<{ value: QueryType; label: string; marker: string }> = [
  { value: "text", label: "文字", marker: "T" },
  { value: "image", label: "图片", marker: "I" },
  { value: "image_text", label: "图文", marker: "T＋I" },
];

const CHANNEL_LABELS: Record<string, string> = {
  native_multimodal: "原生多模态",
  asset_description: "内容描述",
  subject_content: "主体内容",
  scene_theme: "场景主题",
  visual_style: "视觉风格",
  color_composition: "色彩构图",
  mood_atmosphere: "情绪氛围",
  character_state_or_psychology: "人物状态",
  asset_usage: "素材用途",
  target_audience: "目标受众",
  provenance: "来源",
  rights_version_authorship: "版权版本",
};

const ASSET_LABELS: Record<AssetType, string> = {
  image: "图片",
  video_segment: "视频片段",
  markdown_block: "文字段落",
};

const DEMO_RESULTS: SearchResult[] = [
  {
    asset_id: "asset_demo_twilight_01",
    asset_type: "image",
    asset_name: "午后，黄昏将至",
    asset_description:
      "暖金色斜阳穿过街道树冠，女孩在自行车旁停留。画面有动画电影般的叙事感。",
    asset_features: {},
    source_contexts: [
      {
        text: "午后-黄昏：想收集一些日常与旅行交界处的光线，安静，但有故事正在发生。",
        relation_type: "preceding_text",
        text_block_index: 12,
      },
    ],
    source_locator: { block_index: 13 },
    preview_uri:
      "https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?auto=format&fit=crop&w=1200&q=82",
    source_file: {
      source_file_id: "src_demo_moodboard",
      original_file_name: "2026-夏日情绪板.md",
      file_type: "markdown",
      relative_path: "灵感库/视觉参考/2026-夏日情绪板.md",
    },
    score: 0.04428,
    matched_channels: [
      {
        channel: "native_multimodal",
        embedding_type: "native_multimodal",
        rank: 1,
        similarity: 0.936,
        fusion_contribution: 0.01639,
        rrf_contribution: 0.01639,
      },
      {
        channel: "visual_style",
        embedding_type: "visual_style",
        rank: 2,
        similarity: 0.901,
        fusion_contribution: 0.00968,
        rrf_contribution: 0.00968,
      },
    ],
    matched_feature: "日系动画电影感，柔和颗粒",
    matched_reason: "主体、色彩和动画电影质感同时命中",
    rerank_score: 0.94,
    group_kind: null,
    folded_asset_ids: ["asset_demo_twilight_01"],
    available: true,
  },
  {
    asset_id: "asset_demo_field_02",
    asset_type: "video_segment",
    asset_name: "麦田里的归途",
    asset_description:
      "远山与麦田被低角度阳光切开，人物牵马缓慢穿过画面，色彩偏青绿与金黄。",
    asset_features: {},
    source_contexts: [
      {
        text: "这一组的核心不是落日，而是傍晚时人物和环境之间的尺度关系。",
        relation_type: "preceding_text",
      },
    ],
    source_locator: { start_seconds: 42.3, end_seconds: 49.8 },
    preview_uri:
      "https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1200&q=82",
    source_file: {
      source_file_id: "src_demo_reference",
      original_file_name: "田野参考.mp4",
      file_type: "video",
      relative_path: "项目/短片A/参考/田野参考.mp4",
    },
    score: 0.03971,
    matched_channels: [
      {
        channel: "subject_content",
        embedding_type: "subject_content",
        rank: 1,
        similarity: 0.918,
        fusion_contribution: 0.01311,
        rrf_contribution: 0.01311,
      },
      {
        channel: "mood_atmosphere",
        embedding_type: "mood_atmosphere",
        rank: 3,
        similarity: 0.867,
        fusion_contribution: 0.0084,
        rrf_contribution: 0.0084,
      },
    ],
    matched_feature: "人物、马匹、金色田野",
    matched_reason: "命中人物尺度与安静的黄昏氛围",
    rerank_score: 0.88,
    group_kind: "video_segments",
    folded_asset_ids: ["asset_demo_field_02", "asset_demo_field_03"],
    available: true,
  },
  {
    asset_id: "asset_demo_notes_04",
    asset_type: "markdown_block",
    asset_name: "黄昏氛围关键词",
    asset_description: "关于暖色暮光、蓝调时刻、长阴影和克制叙事的创作笔记。",
    asset_features: {},
    source_contexts: [
      {
        text: "黄昏不是橙色滤镜。它更像两个时间系统短暂重叠：街灯开始亮，天空还没有完全暗。",
        relation_type: "self",
        text_block_index: 8,
      },
    ],
    source_locator: { heading: "光线与时间", block_index: 8 },
    preview_uri: null,
    source_file: {
      source_file_id: "src_demo_notes",
      original_file_name: "导演阐述.md",
      file_type: "markdown",
      relative_path: "项目/短片A/导演阐述.md",
    },
    score: 0.03021,
    matched_channels: [
      {
        channel: "asset_description",
        embedding_type: "asset_description",
        rank: 2,
        similarity: 0.892,
        fusion_contribution: 0.0129,
        rrf_contribution: 0.0129,
      },
    ],
    matched_feature: null,
    matched_reason: "文字描述命中蓝调时刻和长阴影",
    rerank_score: null,
    group_kind: null,
    folded_asset_ids: ["asset_demo_notes_04"],
    available: true,
  },
];

const DEMO_RESPONSE: SearchResponse = {
  query: {
    query_type: "text",
    query_text: "蓝紫色黄昏动画场景",
    query_image_url: null,
    query_image_upload_id: null,
    precision_mode: true,
  },
  parsed_query: {
    query_summary: "蓝紫色黄昏中的动画叙事场景",
    dimension_queries: [
      {
        embedding_type: "asset_description",
        query: "蓝紫色黄昏动画场景",
        weight: 0.35,
        source: "text",
        constraint: "match",
      },
      {
        embedding_type: "native_multimodal",
        query: "蓝紫色黄昏动画场景",
        weight: 0.35,
        source: "text",
        constraint: "match",
      },
      {
        embedding_type: "subject_content",
        query: "黄昏中的人物与环境",
        weight: 0.15,
        source: "text",
        constraint: "match",
      },
      {
        embedding_type: "visual_style",
        query: "动画电影质感",
        weight: 0.15,
        source: "text",
        constraint: "match",
      },
    ],
    negative_terms: [],
    parser_mode: "model",
  },
  fusion_method: "weighted_rrf",
  rerank_method: "doubao_seed_2_lite",
  search_engine_version: "search-v1",
  execution_id: "search_exec_demo",
  capsule_id: "search_capsule_demo",
  total: DEMO_RESULTS.length,
  degraded: false,
  degraded_reasons: [],
  timings: { total_ms: 286 },
  results: DEMO_RESULTS,
};

function getPreviewUrl(uri: string | null) {
  if (!uri) return null;
  return /^(https?:|data:image\/)/.test(uri) ? uri : null;
}

function locatorTime(locator: Record<string, unknown>) {
  const raw =
    locator.start_seconds ??
    locator.start_time_seconds ??
    (typeof locator.start_ms === "number" ? locator.start_ms / 1000 : null);
  if (typeof raw !== "number") return null;
  const minutes = Math.floor(raw / 60);
  const seconds = Math.floor(raw % 60)
    .toString()
    .padStart(2, "0");
  return `${minutes}:${seconds}`;
}

function SearchResultCard({
  result,
  index,
  fusionMethod,
}: {
  result: SearchResult;
  index: number;
  fusionMethod: FusionMethod;
}) {
  const [imageFailed, setImageFailed] = useState(false);
  const previewUrl = getPreviewUrl(result.preview_uri);
  const startTime = locatorTime(result.source_locator);
  const context = result.source_contexts.find((item) => item.text)?.text;
  const foldedCount = result.folded_asset_ids?.length ?? 1;

  return (
    <article
      className={`result-card result-card-${index % 4} ${
        result.available ? "" : "result-unavailable"
      }`}
    >
      <div className="result-visual">
        {previewUrl && !imageFailed ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={previewUrl}
            alt={result.asset_name ?? "检索结果预览"}
            onError={() => setImageFailed(true)}
          />
        ) : (
          <div className="visual-placeholder" aria-hidden="true">
            <span>{result.asset_type === "markdown_block" ? "¶" : "C"}</span>
            <i />
          </div>
        )}
        <div className="visual-topline">
          <span>{ASSET_LABELS[result.asset_type]}</span>
          {startTime && <span className="timecode">▶ {startTime}</span>}
        </div>
        <div className="score-stamp">
          <small>
            {fusionMethod === "weighted_rrf" ? "RRF" : "NORM"}
          </small>
          <strong>{result.score.toFixed(4)}</strong>
        </div>
      </div>

      <div className="result-body">
        <div className="result-index">{String(index + 1).padStart(2, "0")}</div>
        <h2>{result.asset_name || "未命名 Asset"}</h2>
        <p className="result-description">{result.asset_description}</p>

        {foldedCount > 1 && (
          <div className="folded-badge">已合并 {foldedCount} 个相邻片段</div>
        )}
        {result.matched_reason && (
          <div className="match-reason">
            <span>为什么命中</span>
            {result.matched_reason}
          </div>
        )}
        {result.matched_feature && (
          <div className="matched-feature">
            <span>命中特征</span>
            {result.matched_feature}
          </div>
        )}

        <div className="channel-list" aria-label="命中通道">
          {result.matched_channels.map((channel) => (
            <div
              className="channel-chip"
              key={`${result.asset_id}-${channel.channel}`}
              title={`排名 ${channel.rank}，融合贡献 ${channel.fusion_contribution.toFixed(5)}`}
            >
              <span>
                {CHANNEL_LABELS[channel.embedding_type] ??
                  channel.embedding_type}
              </span>
              <strong>{channel.similarity.toFixed(3)}</strong>
            </div>
          ))}
        </div>

        {context && (
          <blockquote>
            <span>关联段落</span>
            <p>{context}</p>
          </blockquote>
        )}

        {result.source_file && (
          <footer className="source-footer">
            <div className="file-mark" aria-hidden="true">
              {result.source_file.file_type.slice(0, 2).toUpperCase()}
            </div>
            <div>
              <strong>{result.source_file.original_file_name}</strong>
              <span>{result.source_file.relative_path}</span>
            </div>
          </footer>
        )}
      </div>
    </article>
  );
}

function QueryPlan({ parsed }: { parsed: ParsedQuery | null }) {
  if (!parsed) return null;
  return (
    <section className="query-plan">
      <div>
        <span className="eyebrow">QUERY PLAN</span>
        <strong>{parsed.query_summary}</strong>
        <small>{parsed.parser_mode}</small>
      </div>
      <div className="dimension-strip">
        {parsed.dimension_queries.map((dimension) => (
          <div key={dimension.embedding_type}>
            <span>
              {CHANNEL_LABELS[dimension.embedding_type] ??
                dimension.embedding_type}
            </span>
            <b>{Math.round(dimension.weight * 100)}%</b>
            <small>
              {dimension.source} · {dimension.constraint}
            </small>
          </div>
        ))}
      </div>
      {parsed.negative_terms.length > 0 && (
        <p>排除：{parsed.negative_terms.join("、")}</p>
      )}
    </section>
  );
}

export default function Home() {
  const [activeView, setActiveView] = useState<"search" | "capsules">(
    "search",
  );
  const [queryType, setQueryType] = useState<QueryType>("text");
  const [queryText, setQueryText] = useState("蓝紫色黄昏动画场景");
  const [queryImageUrl, setQueryImageUrl] = useState("");
  const [queryImageFile, setQueryImageFile] = useState<File | null>(null);
  const [workspaceId, setWorkspaceId] = useState("workspace_demo");
  const [createdBy, setCreatedBy] = useState("user_demo");
  const [apiBaseUrl, setApiBaseUrl] = useState(API_BASE_URL);
  const [assetTypes, setAssetTypes] = useState<AssetType[]>([
    "image",
    "video_segment",
  ]);
  const [precisionMode, setPrecisionMode] = useState(true);
  const [fusionMethod, setFusionMethod] =
    useState<FusionMethod>("weighted_rrf");
  const [rerank, setRerank] = useState(true);
  const [saveCapsule, setSaveCapsule] = useState(false);
  const [projectId, setProjectId] = useState("");
  const [sourceFileId, setSourceFileId] = useState("");
  const [fileTypes, setFileTypes] = useState("");
  const [modelVersion, setModelVersion] = useState("");
  const [clusterCapsuleId, setClusterCapsuleId] = useState("");
  const [favoriteOnly, setFavoriteOnly] = useState(false);
  const [response, setResponse] = useState<SearchResponse>(DEMO_RESPONSE);
  const [viewMode, setViewMode] = useState<"demo" | "live">("demo");
  const [capsules, setCapsules] = useState<CapsuleSummary[]>([]);
  const [selectedCapsule, setSelectedCapsule] =
    useState<CapsuleDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const imageQueryEnabled =
    queryType === "image" || queryType === "image_text";
  const textQueryEnabled = queryType === "text" || queryType === "image_text";

  const validationMessage = useMemo(() => {
    if (!workspaceId.trim()) return "请填写 Workspace ID";
    if (textQueryEnabled && !queryText.trim()) return "请输入检索文字";
    if (
      imageQueryEnabled &&
      !queryImageUrl.trim() &&
      queryImageFile === null
    ) {
      return "请上传查询图片，或填写图片 URL";
    }
    return null;
  }, [
    imageQueryEnabled,
    queryImageFile,
    queryImageUrl,
    queryText,
    textQueryEnabled,
    workspaceId,
  ]);

  const endpoint = (path: string) =>
    `${apiBaseUrl.replace(/\/$/, "")}${path}`;

  const readError = async (apiResponse: Response) => {
    const payload = (await apiResponse.json().catch(() => null)) as
      | { detail?: { message?: string } | string }
      | null;
    const detail =
      payload && typeof payload.detail === "object"
        ? payload.detail?.message
        : payload?.detail;
    return detail || `请求失败（${apiResponse.status}）`;
  };

  const uploadImage = async () => {
    if (!queryImageFile) return null;
    const form = new FormData();
    form.set("workspace_id", workspaceId.trim());
    form.set("file", queryImageFile);
    const uploadResponse = await fetch(endpoint("/api/v1/query-images"), {
      method: "POST",
      body: form,
    });
    if (!uploadResponse.ok) throw new Error(await readError(uploadResponse));
    return (await uploadResponse.json()) as { upload_id: string };
  };

  const submitSearch = async (event: FormEvent) => {
    event.preventDefault();
    if (validationMessage) {
      setError(validationMessage);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const upload = imageQueryEnabled ? await uploadImage() : null;
      const apiResponse = await fetch(endpoint("/api/v1/search"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          workspace_id: workspaceId.trim(),
          created_by: createdBy.trim(),
          query_type: queryType,
          query_text: textQueryEnabled ? queryText.trim() : null,
          query_image_url:
            imageQueryEnabled && !upload ? queryImageUrl.trim() : null,
          query_image_upload_id: upload?.upload_id ?? null,
          precision_mode: precisionMode,
          fusion_method: fusionMethod,
          rerank: rerank ? "doubao_seed_2_lite" : "off",
          save_capsule: saveCapsule,
          filters: {
            project_id: projectId.trim() || null,
            asset_type: assetTypes,
            file_type: fileTypes
              .split(",")
              .map((item) => item.trim())
              .filter(Boolean),
            source_file_id: sourceFileId.trim()
              ? [sourceFileId.trim()]
              : [],
            embedding_model_version: modelVersion.trim()
              ? [modelVersion.trim()]
              : [],
            favorite: favoriteOnly ? true : null,
            cluster_capsule_id: clusterCapsuleId.trim() || null,
          },
          top_k: 20,
        }),
      });
      if (!apiResponse.ok) throw new Error(await readError(apiResponse));
      setResponse((await apiResponse.json()) as SearchResponse);
      setViewMode("live");
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "无法连接搜索服务，请检查 API 地址。",
      );
    } finally {
      setLoading(false);
    }
  };

  const loadCapsules = async () => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({
        workspace_id: workspaceId,
        created_by: createdBy,
      });
      const apiResponse = await fetch(
        endpoint(`/api/v1/search-capsules?${params}`),
      );
      if (!apiResponse.ok) throw new Error(await readError(apiResponse));
      const payload = (await apiResponse.json()) as {
        items: CapsuleSummary[];
      };
      setCapsules(payload.items);
      setViewMode("live");
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Search Capsule 加载失败",
      );
    } finally {
      setLoading(false);
    }
  };

  const openCapsule = async (capsuleId: string) => {
    setLoading(true);
    try {
      const params = new URLSearchParams({
        workspace_id: workspaceId,
        created_by: createdBy,
      });
      const apiResponse = await fetch(
        endpoint(`/api/v1/search-capsules/${capsuleId}?${params}`),
      );
      if (!apiResponse.ok) throw new Error(await readError(apiResponse));
      setSelectedCapsule((await apiResponse.json()) as CapsuleDetail);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "打开失败");
    } finally {
      setLoading(false);
    }
  };

  const refreshCapsule = async (capsuleId: string) => {
    setLoading(true);
    try {
      const params = new URLSearchParams({
        workspace_id: workspaceId,
        created_by: createdBy,
      });
      const apiResponse = await fetch(
        endpoint(`/api/v1/search-capsules/${capsuleId}/refresh?${params}`),
        { method: "POST" },
      );
      if (!apiResponse.ok) throw new Error(await readError(apiResponse));
      setResponse((await apiResponse.json()) as SearchResponse);
      setActiveView("search");
      setViewMode("live");
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "刷新失败");
    } finally {
      setLoading(false);
    }
  };

  const toggleFavorite = async (capsule: CapsuleSummary) => {
    const params = new URLSearchParams({
      workspace_id: workspaceId,
      created_by: createdBy,
    });
    const apiResponse = await fetch(
      endpoint(`/api/v1/search-capsules/${capsule.capsule_id}?${params}`),
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ is_favorite: !capsule.is_favorite }),
      },
    );
    if (!apiResponse.ok) {
      setError(await readError(apiResponse));
      return;
    }
    await loadCapsules();
  };

  const deleteCapsule = async (capsuleId: string) => {
    if (!window.confirm("确认删除这个 Search Capsule？历史快照也会一并删除。")) {
      return;
    }
    const params = new URLSearchParams({
      workspace_id: workspaceId,
      created_by: createdBy,
    });
    const apiResponse = await fetch(
      endpoint(`/api/v1/search-capsules/${capsuleId}?${params}`),
      { method: "DELETE" },
    );
    if (!apiResponse.ok) {
      setError(await readError(apiResponse));
      return;
    }
    setSelectedCapsule(null);
    await loadCapsules();
  };

  const handleQueryKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  };

  const toggleAssetType = (assetType: AssetType) => {
    setAssetTypes((current) =>
      current.includes(assetType)
        ? current.filter((item) => item !== assetType)
        : [...current, assetType],
    );
  };

  return (
    <main className="app-shell">
      <ProductTopbar
        active="search"
        connection={viewMode}
        status={viewMode === "live" ? "实时服务" : "演示数据"}
        workspace={workspaceId || "未选择 Workspace"}
      />

      {activeView === "search" ? (
        <div className="workspace" id="top">
          <aside className="query-panel">
            <div className="query-heading">
              <span className="eyebrow">SEARCH / 01</span>
              <h1>
                搜到你记得的
                <br />
                那一幕。
              </h1>
              <p>Query Parser、12 路召回、融合、重排和结果折叠一次完成。</p>
            </div>

            <form onSubmit={submitSearch}>
              <fieldset className="query-type-switcher">
                <legend>查询方式</legend>
                {QUERY_TYPES.map((item) => (
                  <label
                    className={queryType === item.value ? "active" : ""}
                    key={item.value}
                  >
                    <input
                      type="radio"
                      name="query-type"
                      checked={queryType === item.value}
                      onChange={() => setQueryType(item.value)}
                    />
                    <span>{item.marker}</span>
                    {item.label}
                  </label>
                ))}
              </fieldset>

              {textQueryEnabled && (
                <label className="field-group">
                  <span>你记得什么？</span>
                  <textarea
                    value={queryText}
                    onChange={(event) => setQueryText(event.target.value)}
                    onKeyDown={handleQueryKeyDown}
                    placeholder="例如：保持构图，更像黄昏，排除文字水印…"
                    rows={4}
                  />
                  <small>⌘ Enter 快速检索</small>
                </label>
              )}

              {imageQueryEnabled && (
                <div className="image-inputs">
                  <label className="upload-field">
                    <span>{queryImageFile?.name ?? "上传查询图片"}</span>
                    <input
                      type="file"
                      accept="image/*"
                      onChange={(event) =>
                        setQueryImageFile(event.target.files?.[0] ?? null)
                      }
                    />
                  </label>
                  <label className="field-group compact">
                    <span>或使用图片 URL</span>
                    <input
                      type="url"
                      value={queryImageUrl}
                      onChange={(event) => setQueryImageUrl(event.target.value)}
                      placeholder="https://…/reference.jpg"
                      disabled={queryImageFile !== null}
                    />
                  </label>
                </div>
              )}

              <div className="search-options">
                <label>
                  <input
                    type="checkbox"
                    checked={precisionMode}
                    onChange={(event) => setPrecisionMode(event.target.checked)}
                  />
                  <span />
                  精搜模式
                </label>
                <label>
                  <input
                    type="checkbox"
                    checked={rerank}
                    onChange={(event) => setRerank(event.target.checked)}
                  />
                  <span />
                  豆包重排
                </label>
                <label>
                  <input
                    type="checkbox"
                    checked={saveCapsule}
                    onChange={(event) => setSaveCapsule(event.target.checked)}
                  />
                  <span />
                  收藏本次
                </label>
              </div>

              <label className="select-field">
                融合算法
                <select
                  value={fusionMethod}
                  onChange={(event) =>
                    setFusionMethod(event.target.value as FusionMethod)
                  }
                >
                  <option value="weighted_rrf">Weighted RRF</option>
                  <option value="normalized_weighted_similarity">
                    Normalized Similarity
                  </option>
                </select>
              </label>

              <fieldset className="asset-filters">
                <legend>素材类型</legend>
                {(Object.keys(ASSET_LABELS) as AssetType[]).map((assetType) => (
                  <label key={assetType}>
                    <input
                      type="checkbox"
                      checked={assetTypes.includes(assetType)}
                      onChange={() => toggleAssetType(assetType)}
                    />
                    <span aria-hidden="true" />
                    {ASSET_LABELS[assetType]}
                  </label>
                ))}
              </fieldset>

              <details className="advanced-filters">
                <summary>高级过滤</summary>
                <div>
                  <input
                    value={projectId}
                    onChange={(event) => setProjectId(event.target.value)}
                    placeholder="Project ID"
                  />
                  <input
                    value={sourceFileId}
                    onChange={(event) => setSourceFileId(event.target.value)}
                    placeholder="Source File ID"
                  />
                  <input
                    value={fileTypes}
                    onChange={(event) => setFileTypes(event.target.value)}
                    placeholder="文件类型，逗号分隔"
                  />
                  <input
                    value={modelVersion}
                    onChange={(event) => setModelVersion(event.target.value)}
                    placeholder="Embedding 模型版本"
                  />
                  <input
                    value={clusterCapsuleId}
                    onChange={(event) =>
                      setClusterCapsuleId(event.target.value)
                    }
                    placeholder="Cluster Capsule ID"
                  />
                  <label>
                    <input
                      type="checkbox"
                      checked={favoriteOnly}
                      onChange={(event) =>
                        setFavoriteOnly(event.target.checked)
                      }
                    />
                    仅收藏素材
                  </label>
                </div>
              </details>

              <button className="search-button" type="submit" disabled={loading}>
                <span>{loading ? "正在检索…" : "开始检索"}</span>
                <b aria-hidden="true">↗</b>
              </button>

              {error && (
                <div className="error-message" role="alert">
                  <strong>请求未完成</strong>
                  <span>{error}</span>
                  <button
                    type="button"
                    onClick={() => {
                      setResponse(DEMO_RESPONSE);
                      setViewMode("demo");
                      setError(null);
                    }}
                  >
                    查看演示结果
                  </button>
                </div>
              )}
            </form>

            <details className="connection-settings">
              <summary>连接设置</summary>
              <label>
                Workspace ID
                <input
                  value={workspaceId}
                  onChange={(event) => setWorkspaceId(event.target.value)}
                />
              </label>
              <label>
                User ID
                <input
                  value={createdBy}
                  onChange={(event) => setCreatedBy(event.target.value)}
                />
              </label>
              <label>
                Search API
                <input
                  value={apiBaseUrl}
                  onChange={(event) => setApiBaseUrl(event.target.value)}
                />
              </label>
            </details>
          </aside>

          <section className="results-panel">
            <div className="results-header">
              <div>
                <span className="eyebrow">RESULTS / 02</span>
                <h2>
                  {response.total}
                  <small> 个相关结果</small>
                </h2>
              </div>
              <div className="results-meta">
                <span>
                  {Math.round(response.timings.total_ms)} ms
                </span>
                <span>
                  {response.fusion_method === "weighted_rrf"
                    ? "Weighted RRF"
                    : "Normalized"}
                </span>
                <span>
                  {response.rerank_method === "off" ? "未重排" : "Seed 重排"}
                </span>
              </div>
            </div>

            <QueryPlan parsed={response.parsed_query} />

            {response.degraded && (
              <div className="degraded-banner">
                <strong>部分链路已降级</strong>
                <span>{response.degraded_reasons.join("；")}</span>
              </div>
            )}

            {response.results.length > 0 ? (
              <div className="results-grid">
                {response.results.map((result, index) => (
                  <SearchResultCard
                    result={result}
                    index={index}
                    fusionMethod={response.fusion_method}
                    key={result.asset_id}
                  />
                ))}
              </div>
            ) : (
              <div className="empty-state">
                <span>0</span>
                <h2>没有找到相似素材</h2>
                <p>尝试减少过滤条件，或换一种更具体的场景描述。</p>
              </div>
            )}
          </section>
        </div>
      ) : (
        <section className="capsule-page" id="top">
          <header>
            <div>
              <span className="eyebrow">MEMORY / 03</span>
              <h1>Search Capsules</h1>
              <p>最近 10 次成功检索与全部收藏；历史快照和刷新结果彼此独立。</p>
            </div>
            <button onClick={() => void loadCapsules()} disabled={loading}>
              {loading ? "刷新中…" : "刷新列表"}
            </button>
          </header>
          {error && <div className="degraded-banner">{error}</div>}
          <div className="capsule-layout">
            <div className="capsule-list">
              {capsules.map((capsule) => (
                <article
                  className={
                    selectedCapsule?.capsule_id === capsule.capsule_id
                      ? "active"
                      : ""
                  }
                  key={capsule.capsule_id}
                >
                  <button
                    className="capsule-open"
                    onClick={() => void openCapsule(capsule.capsule_id)}
                  >
                    <small>
                      {capsule.query_type} · {capsule.result_count} results
                    </small>
                    <strong>{capsule.query_summary}</strong>
                    <span>
                      {new Date(capsule.last_used_at).toLocaleString("zh-CN")}
                    </span>
                  </button>
                  <button
                    className="favorite-button"
                    onClick={() => void toggleFavorite(capsule)}
                    aria-label={capsule.is_favorite ? "取消收藏" : "收藏"}
                  >
                    {capsule.is_favorite ? "★" : "☆"}
                  </button>
                </article>
              ))}
              {!loading && capsules.length === 0 && (
                <div className="empty-capsules">暂无 Search Capsule</div>
              )}
            </div>
            <div className="capsule-detail">
              {selectedCapsule ? (
                <>
                  <header>
                    <div>
                      <small>历史快照</small>
                      <h2>{selectedCapsule.query_summary}</h2>
                      <span>
                        {selectedCapsule.executions.length} 次执行 ·{" "}
                        {selectedCapsule.latest_snapshot.results.length} 个结果
                      </span>
                    </div>
                    <div>
                      <button
                        onClick={() =>
                          void refreshCapsule(selectedCapsule.capsule_id)
                        }
                      >
                        用最新索引刷新
                      </button>
                      <button
                        className="danger"
                        onClick={() =>
                          void deleteCapsule(selectedCapsule.capsule_id)
                        }
                      >
                        删除
                      </button>
                    </div>
                  </header>
                  <QueryPlan parsed={selectedCapsule.parsed_query} />
                  <div className="results-grid compact-grid">
                    {selectedCapsule.latest_snapshot.results.map(
                      (result, index) => (
                        <SearchResultCard
                          result={result}
                          index={index}
                          fusionMethod={selectedCapsule.fusion_method}
                          key={`${selectedCapsule.capsule_id}-${result.asset_id}`}
                        />
                      ),
                    )}
                  </div>
                </>
              ) : (
                <div className="empty-state">
                  <span>◎</span>
                  <h2>选择一条检索记忆</h2>
                  <p>打开当时的结果快照，或使用最新索引重新执行。</p>
                </div>
              )}
            </div>
          </div>
        </section>
      )}
    </main>
  );
}
