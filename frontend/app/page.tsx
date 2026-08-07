"use client";

import {
  FormEvent,
  KeyboardEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { ProductTopbar } from "./components/DemoShell";
import { endpoint } from "./lib/api";
import { useWorkspaceSelection, WorkspaceSelect } from "./lib/workspaces";

type QueryType = "text" | "image" | "image_text";
type AssetType = "image" | "video_segment" | "markdown_block" | "text_block";
type FusionMethod = "weighted_rrf" | "normalized_weighted_similarity";
type EmbeddingType =
  | "native_multimodal"
  | "asset_description"
  | "subject_content"
  | "scene_theme"
  | "visual_style"
  | "color_composition"
  | "mood_atmosphere"
  | "character_state_or_psychology"
  | "asset_usage"
  | "target_audience"
  | "provenance"
  | "rights_version_authorship";

type DimensionQuery = {
  embedding_type: string;
  query: string;
  weight: number;
  source: "text" | "image" | "joint";
};

type ParsedQuery = {
  dimension_queries: DimensionQuery[];
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

type ClusterSearchResult = {
  cluster_capsule_id: string;
  cluster_run_id: string;
  embedding_type: string;
  name: string;
  description: string;
  keywords: string[];
  common_features: string[];
  member_count: number;
  average_membership_probability: number;
  medoid_asset_id: string | null;
  representative_asset_ids: string[];
  matched_asset_ids: string[];
  matched_asset_count: number;
  score: number;
};

type SearchResponse = {
  query: {
    query_type: QueryType;
    query_text: string | null;
    query_image_url: string | null;
    query_image_upload_id: string | null;
    embedding_types: EmbeddingType[];
  };
  parsed_query: ParsedQuery | null;
  fusion_method: FusionMethod;
  rerank_method: "off" | "doubao_seed_2_lite";
  search_engine_version: string;
  execution_id: string | null;
  capsule_id: string | null;
  total: number;
  asset_total: number;
  cluster_total: number;
  degraded: boolean;
  degraded_reasons: string[];
  timings: {
    query_enhancement_ms: number;
    total_ms: number;
  };
  assets: SearchResult[];
  clusters: ClusterSearchResult[];
  results: SearchResult[];
};

type CapsuleSummary = {
  capsule_id: string;
  query_type: QueryType;
  query_text: string | null;
  query_image_uri: string | null;
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

const QUERY_TYPES: Array<{ value: QueryType; label: string; marker: string }> = [
  { value: "text", label: "文字", marker: "T" },
  { value: "image", label: "图片", marker: "I" },
  { value: "image_text", label: "图文", marker: "T＋I" },
];

const CHANNEL_LABELS: Record<string, string> = {
  native_multimodal: "原始内容",
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

const SEARCH_DIMENSIONS = Object.entries(CHANNEL_LABELS).map(
  ([value, label]) => ({ value: value as EmbeddingType, label }),
);

const ASSET_LABELS: Record<AssetType, string> = {
  image: "图片",
  video_segment: "视频片段",
  markdown_block: "Markdown 段落",
  text_block: "纯文本块",
};

const VISUAL_ASSET_TYPES = new Set<AssetType>(["image", "video_segment"]);
const VISUAL_ONLY_DIMENSIONS = new Set<EmbeddingType>([
  "visual_style",
  "color_composition",
]);

function dimensionSupportCount(
  embeddingType: EmbeddingType,
  targetAssetTypes: AssetType[],
) {
  if (!VISUAL_ONLY_DIMENSIONS.has(embeddingType)) {
    return targetAssetTypes.length;
  }
  return targetAssetTypes.filter((assetType) =>
    VISUAL_ASSET_TYPES.has(assetType),
  ).length;
}

function dimensionSupportsTargets(
  embeddingType: EmbeddingType,
  targetAssetTypes: AssetType[],
) {
  return dimensionSupportCount(embeddingType, targetAssetTypes) > 0;
}

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

const DEMO_CLUSTERS: ClusterSearchResult[] = [
  {
    cluster_capsule_id: "cc_demo_twilight",
    cluster_run_id: "run_demo_mood",
    embedding_type: "mood_atmosphere",
    name: "蓝紫与暖金交界的黄昏",
    description:
      "这个簇聚合了蓝调时刻、暖金斜阳与克制叙事感的图片和视频片段，主要差异来自人物是否出现以及环境尺度。",
    keywords: ["蓝调时刻", "暖金斜阳", "安静叙事"],
    common_features: ["黄昏", "动画电影感"],
    member_count: 18,
    average_membership_probability: 0.91,
    medoid_asset_id: "asset_demo_twilight_01",
    representative_asset_ids: [
      "asset_demo_twilight_01",
      "asset_demo_field_02",
    ],
    matched_asset_ids: ["asset_demo_twilight_01", "asset_demo_field_02"],
    matched_asset_count: 2,
    score: 0.94,
  },
];

const DEMO_RESPONSE: SearchResponse = {
  query: {
    query_type: "text",
    query_text: "蓝紫色黄昏动画场景",
    query_image_url: null,
    query_image_upload_id: null,
    embedding_types: [
      "asset_description",
      "native_multimodal",
      "subject_content",
      "visual_style",
    ],
  },
  parsed_query: {
    dimension_queries: [
      {
        embedding_type: "asset_description",
        query: "蓝紫色黄昏时分的动画场景，呈现完整画面内容与环境氛围",
        weight: 0.35,
        source: "text",
      },
      {
        embedding_type: "native_multimodal",
        query: "蓝紫色黄昏动画场景",
        weight: 0.35,
        source: "text",
      },
      {
        embedding_type: "subject_content",
        query: "黄昏场景中的人物、环境与主体内容",
        weight: 0.15,
        source: "text",
      },
      {
        embedding_type: "visual_style",
        query: "蓝紫色调的动画电影视觉风格",
        weight: 0.15,
        source: "text",
      },
    ],
  },
  fusion_method: "weighted_rrf",
  rerank_method: "doubao_seed_2_lite",
  search_engine_version: "search-v1",
  execution_id: "search_exec_demo",
  capsule_id: "search_capsule_demo",
  total: DEMO_RESULTS.length,
  asset_total: DEMO_RESULTS.length,
  cluster_total: DEMO_CLUSTERS.length,
  degraded: false,
  degraded_reasons: [],
  timings: { query_enhancement_ms: 74, total_ms: 286 },
  assets: DEMO_RESULTS,
  clusters: DEMO_CLUSTERS,
  results: DEMO_RESULTS,
};

const EMPTY_RESPONSE: SearchResponse = {
  ...DEMO_RESPONSE,
  query: {
    query_type: "text",
    query_text: null,
    query_image_url: null,
    query_image_upload_id: null,
    embedding_types: ["native_multimodal"],
  },
  parsed_query: null,
  fusion_method: "weighted_rrf",
  rerank_method: "doubao_seed_2_lite",
  search_engine_version: "search-v1",
  execution_id: null,
  capsule_id: null,
  total: 0,
  asset_total: 0,
  cluster_total: 0,
  degraded: false,
  degraded_reasons: [],
  timings: { query_enhancement_ms: 0, total_ms: 0 },
  assets: [],
  clusters: [],
  results: [],
};

function getPreviewUrl(
  result: SearchResult,
  workspaceId: string,
) {
  if (result.preview_uri && /^(https?:|data:image\/)/.test(result.preview_uri)) {
    return result.preview_uri;
  }
  if (result.asset_type !== "image") return null;
  return endpoint(`/api/v1/assets/${encodeURIComponent(
    result.asset_id,
  )}/thumbnail?workspace_id=${encodeURIComponent(workspaceId)}`);
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
  workspaceId,
}: {
  result: SearchResult;
  index: number;
  fusionMethod: FusionMethod;
  workspaceId: string;
}) {
  const [imageFailed, setImageFailed] = useState(false);
  const previewUrl = getPreviewUrl(result, workspaceId);
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
            <span>
              {result.asset_type === "markdown_block" ||
              result.asset_type === "text_block"
                ? "¶"
                : "C"}
            </span>
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

function ClusterResultCard({
  cluster,
  index,
}: {
  cluster: ClusterSearchResult;
  index: number;
}) {
  const label =
    CHANNEL_LABELS[cluster.embedding_type] ?? cluster.embedding_type;
  return (
    <article className={`cluster-search-card cluster-search-card-${index % 4}`}>
      <div className="cluster-search-visual" aria-hidden="true">
        <div className="cluster-search-orbit">
          {Array.from({
            length: Math.min(9, Math.max(3, cluster.matched_asset_count + 2)),
          }).map((_, dotIndex) => (
            <i key={dotIndex} />
          ))}
          <strong>{cluster.member_count}</strong>
          <span>ASSETS</span>
        </div>
        <div className="cluster-search-score">
          <small>CLUSTER SCORE</small>
          <strong>{cluster.score.toFixed(3)}</strong>
        </div>
      </div>
      <div className="cluster-search-body">
        <header>
          <span>{String(index + 1).padStart(2, "0")}</span>
          <small>{label}</small>
        </header>
        <h2>{cluster.name}</h2>
        <p>{cluster.description}</p>
        <div className="cluster-match-summary">
          <strong>{cluster.matched_asset_count}</strong>
          <span>个当前命中素材落在此簇</span>
          <em>
            平均成员置信度{" "}
            {(cluster.average_membership_probability * 100).toFixed(0)}%
          </em>
        </div>
        <div className="cluster-keywords">
          {Array.from(
            new Set([...cluster.keywords, ...cluster.common_features]),
          )
            .slice(0, 6)
            .map((keyword) => (
              <span key={keyword}>{keyword}</span>
            ))}
        </div>
        <footer>
          <code>{cluster.cluster_capsule_id}</code>
          <a
            href={`/clusters?cluster_run_id=${encodeURIComponent(
              cluster.cluster_run_id,
            )}&cluster_capsule_id=${encodeURIComponent(
              cluster.cluster_capsule_id,
            )}`}
            target="_blank"
            rel="noopener noreferrer"
          >
            打开簇详情 ↗
          </a>
        </footer>
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
      </div>
      <div className="dimension-strip">
        {parsed.dimension_queries.map((dimension) => (
          <div key={dimension.embedding_type}>
            <header>
              <span>
                {CHANNEL_LABELS[dimension.embedding_type] ??
                  dimension.embedding_type}
              </span>
              <b>{Math.round(dimension.weight * 100)}%</b>
            </header>
            <p>{dimension.query}</p>
            <small>source · {dimension.source}</small>
          </div>
        ))}
      </div>
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
  const {
    workspaceId,
    workspaces,
    loading: workspacesLoading,
    ready: workspaceReady,
    setWorkspaceId,
  } = useWorkspaceSelection();
  const [createdBy, setCreatedBy] = useState("user_demo");
  const [assetTypes, setAssetTypes] = useState<AssetType[]>([
    "image",
    "video_segment",
  ]);
  const [embeddingTypes, setEmbeddingTypes] = useState<EmbeddingType[]>([
    "native_multimodal",
  ]);
  const [fusionMethod, setFusionMethod] =
    useState<FusionMethod>("weighted_rrf");
  const [rerank, setRerank] = useState(false);
  const [saveCapsule, setSaveCapsule] = useState(false);
  const [projectId, setProjectId] = useState("");
  const [sourceFileId, setSourceFileId] = useState("");
  const [fileTypes, setFileTypes] = useState("");
  const [modelName, setModelName] = useState("");
  const [clusterCapsuleId, setClusterCapsuleId] = useState("");
  const [favoriteOnly, setFavoriteOnly] = useState(false);
  const [response, setResponse] = useState<SearchResponse>(EMPTY_RESPONSE);
  const [resultSet, setResultSet] = useState<"assets" | "clusters">(
    "assets",
  );
  const [viewMode, setViewMode] = useState<"demo" | "live">("live");
  const [capsules, setCapsules] = useState<CapsuleSummary[]>([]);
  const [selectedCapsule, setSelectedCapsule] =
    useState<CapsuleDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const workspaceIdRef = useRef(workspaceId);
  const requestControllersRef = useRef(new Set<AbortController>());

  useEffect(() => {
    workspaceIdRef.current = workspaceId;
    for (const controller of requestControllersRef.current) {
      controller.abort();
    }
    requestControllersRef.current.clear();
  }, [workspaceId]);

  useEffect(
    () => () => {
      for (const controller of requestControllersRef.current) {
        controller.abort();
      }
    },
    [],
  );

  const startWorkspaceRequest = () => {
    const controller = new AbortController();
    requestControllersRef.current.add(controller);
    return controller;
  };

  const finishWorkspaceRequest = (controller: AbortController) => {
    requestControllersRef.current.delete(controller);
  };

  const isCurrentWorkspace = (requestWorkspaceId: string) =>
    workspaceIdRef.current === requestWorkspaceId;

  const handleWorkspaceChange = (nextWorkspaceId: string) => {
    if (!nextWorkspaceId || nextWorkspaceId === workspaceIdRef.current) return;
    // Update the ref synchronously so a response that resolves during the
    // React state transition cannot paint data from the previous workspace.
    workspaceIdRef.current = nextWorkspaceId;
    for (const controller of requestControllersRef.current) {
      controller.abort();
    }
    requestControllersRef.current.clear();
    setResponse(EMPTY_RESPONSE);
    setCapsules([]);
    setSelectedCapsule(null);
    setError(null);
    setLoading(false);
    setResultSet("assets");
    setViewMode("live");
    setWorkspaceId(nextWorkspaceId);
  };

  const imageQueryEnabled =
    queryType === "image" || queryType === "image_text";
  const textQueryEnabled = queryType === "text" || queryType === "image_text";

  const validationMessage = useMemo(() => {
    if (!workspaceReady) return "正在加载工作空间";
    if (!workspaceId.trim()) return "请填写 Workspace ID";
    if (assetTypes.length === 0) return "请至少选择一种目标素材类型";
    if (embeddingTypes.length === 0) return "请至少选择一个检索维度";
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
    assetTypes,
    imageQueryEnabled,
    embeddingTypes,
    queryImageFile,
    queryImageUrl,
    queryText,
    textQueryEnabled,
    workspaceId,
    workspaceReady,
  ]);

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

  const uploadImage = async (
    requestWorkspaceId: string,
    signal: AbortSignal,
  ) => {
    if (!queryImageFile) return null;
    const form = new FormData();
    form.set("workspace_id", requestWorkspaceId);
    form.set("file", queryImageFile);
    const uploadResponse = await fetch(endpoint("/api/v1/query-images"), {
      method: "POST",
      body: form,
      signal,
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
    const requestWorkspaceId = workspaceId.trim();
    const controller = startWorkspaceRequest();
    setLoading(true);
    setError(null);
    try {
      const upload = imageQueryEnabled
        ? await uploadImage(requestWorkspaceId, controller.signal)
        : null;
      const apiResponse = await fetch(endpoint("/api/v1/search"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          workspace_id: requestWorkspaceId,
          created_by: createdBy.trim(),
          query_type: queryType,
          query_text: textQueryEnabled ? queryText.trim() : null,
          query_image_url:
            imageQueryEnabled && !upload ? queryImageUrl.trim() : null,
          query_image_upload_id: upload?.upload_id ?? null,
          embedding_types: embeddingTypes,
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
            model_name: modelName.trim()
              ? [modelName.trim()]
              : [],
            favorite: favoriteOnly ? true : null,
            cluster_capsule_id: clusterCapsuleId.trim() || null,
          },
          top_k: 20,
        }),
      });
      if (!apiResponse.ok) throw new Error(await readError(apiResponse));
      const nextResponse = (await apiResponse.json()) as SearchResponse;
      if (isCurrentWorkspace(requestWorkspaceId)) {
        setResponse(nextResponse);
        setResultSet("assets");
        setViewMode("live");
      }
    } catch (requestError) {
      if (isCurrentWorkspace(requestWorkspaceId) && !controller.signal.aborted) {
        setError(
          requestError instanceof Error
            ? requestError.message
            : "无法连接搜索服务，请检查 API 地址。",
        );
      }
    } finally {
      finishWorkspaceRequest(controller);
      if (isCurrentWorkspace(requestWorkspaceId)) setLoading(false);
    }
  };

  const loadCapsules = async (
    requestWorkspaceId = workspaceId,
    requestCreatedBy = createdBy,
  ) => {
    const controller = startWorkspaceRequest();
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({
        workspace_id: requestWorkspaceId,
        created_by: requestCreatedBy,
      });
      const apiResponse = await fetch(
        endpoint(`/api/v1/search-capsules?${params}`),
        { signal: controller.signal },
      );
      if (!apiResponse.ok) throw new Error(await readError(apiResponse));
      const payload = (await apiResponse.json()) as {
        items: CapsuleSummary[];
      };
      if (isCurrentWorkspace(requestWorkspaceId)) {
        setCapsules(payload.items);
        setViewMode("live");
      }
    } catch (requestError) {
      if (isCurrentWorkspace(requestWorkspaceId) && !controller.signal.aborted) {
        setError(
          requestError instanceof Error
            ? requestError.message
            : "Search Capsule 加载失败",
        );
      }
    } finally {
      finishWorkspaceRequest(controller);
      if (isCurrentWorkspace(requestWorkspaceId)) setLoading(false);
    }
  };

  const openCapsule = async (capsuleId: string) => {
    const requestWorkspaceId = workspaceId;
    const requestCreatedBy = createdBy;
    const controller = startWorkspaceRequest();
    setLoading(true);
    try {
      const params = new URLSearchParams({
        workspace_id: requestWorkspaceId,
        created_by: requestCreatedBy,
      });
      const apiResponse = await fetch(
        endpoint(`/api/v1/search-capsules/${capsuleId}?${params}`),
        { signal: controller.signal },
      );
      if (!apiResponse.ok) throw new Error(await readError(apiResponse));
      const capsule = (await apiResponse.json()) as CapsuleDetail;
      if (isCurrentWorkspace(requestWorkspaceId)) setSelectedCapsule(capsule);
    } catch (requestError) {
      if (isCurrentWorkspace(requestWorkspaceId) && !controller.signal.aborted) {
        setError(requestError instanceof Error ? requestError.message : "打开失败");
      }
    } finally {
      finishWorkspaceRequest(controller);
      if (isCurrentWorkspace(requestWorkspaceId)) setLoading(false);
    }
  };

  const refreshCapsule = async (capsuleId: string) => {
    const requestWorkspaceId = workspaceId;
    const requestCreatedBy = createdBy;
    const controller = startWorkspaceRequest();
    setLoading(true);
    try {
      const params = new URLSearchParams({
        workspace_id: requestWorkspaceId,
        created_by: requestCreatedBy,
      });
      const apiResponse = await fetch(
        endpoint(`/api/v1/search-capsules/${capsuleId}/refresh?${params}`),
        { method: "POST", signal: controller.signal },
      );
      if (!apiResponse.ok) throw new Error(await readError(apiResponse));
      const nextResponse = (await apiResponse.json()) as SearchResponse;
      if (isCurrentWorkspace(requestWorkspaceId)) {
        setResponse(nextResponse);
        setResultSet("assets");
        setActiveView("search");
        setViewMode("live");
      }
    } catch (requestError) {
      if (isCurrentWorkspace(requestWorkspaceId) && !controller.signal.aborted) {
        setError(requestError instanceof Error ? requestError.message : "刷新失败");
      }
    } finally {
      finishWorkspaceRequest(controller);
      if (isCurrentWorkspace(requestWorkspaceId)) setLoading(false);
    }
  };

  const toggleFavorite = async (capsule: CapsuleSummary) => {
    const requestWorkspaceId = workspaceId;
    const requestCreatedBy = createdBy;
    const controller = startWorkspaceRequest();
    const params = new URLSearchParams({
      workspace_id: requestWorkspaceId,
      created_by: requestCreatedBy,
    });
    setLoading(true);
    try {
      const apiResponse = await fetch(
        endpoint(`/api/v1/search-capsules/${capsule.capsule_id}?${params}`),
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ is_favorite: !capsule.is_favorite }),
          signal: controller.signal,
        },
      );
      if (!apiResponse.ok) throw new Error(await readError(apiResponse));
      if (isCurrentWorkspace(requestWorkspaceId)) {
        await loadCapsules(requestWorkspaceId, requestCreatedBy);
      }
    } catch (requestError) {
      if (isCurrentWorkspace(requestWorkspaceId) && !controller.signal.aborted) {
        setError(requestError instanceof Error ? requestError.message : "更新失败");
      }
    } finally {
      finishWorkspaceRequest(controller);
      if (isCurrentWorkspace(requestWorkspaceId)) setLoading(false);
    }
  };

  const deleteCapsule = async (capsuleId: string) => {
    if (!window.confirm("确认删除这个 Search Capsule？历史快照也会一并删除。")) {
      return;
    }
    const requestWorkspaceId = workspaceId;
    const requestCreatedBy = createdBy;
    const controller = startWorkspaceRequest();
    const params = new URLSearchParams({
      workspace_id: requestWorkspaceId,
      created_by: requestCreatedBy,
    });
    setLoading(true);
    try {
      const apiResponse = await fetch(
        endpoint(`/api/v1/search-capsules/${capsuleId}?${params}`),
        { method: "DELETE", signal: controller.signal },
      );
      if (!apiResponse.ok) throw new Error(await readError(apiResponse));
      if (isCurrentWorkspace(requestWorkspaceId)) {
        setSelectedCapsule(null);
        await loadCapsules(requestWorkspaceId, requestCreatedBy);
      }
    } catch (requestError) {
      if (isCurrentWorkspace(requestWorkspaceId) && !controller.signal.aborted) {
        setError(requestError instanceof Error ? requestError.message : "删除失败");
      }
    } finally {
      finishWorkspaceRequest(controller);
      if (isCurrentWorkspace(requestWorkspaceId)) setLoading(false);
    }
  };

  const handleQueryKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  };

  const toggleAssetType = (assetType: AssetType) => {
    const next = assetTypes.includes(assetType)
      ? assetTypes.filter((item) => item !== assetType)
      : [...assetTypes, assetType];
    if (next.length === 0) return;
    setAssetTypes(next);
    setEmbeddingTypes((current) => {
      const compatible = current.filter((embeddingType) =>
        dimensionSupportsTargets(embeddingType, next),
      );
      return compatible.length > 0 ? compatible : ["native_multimodal"];
    });
  };
  const toggleEmbeddingType = (embeddingType: EmbeddingType) => {
    if (!dimensionSupportsTargets(embeddingType, assetTypes)) return;
    setEmbeddingTypes((current) => {
      if (current.includes(embeddingType)) {
        return current.length === 1
          ? current
          : current.filter((item) => item !== embeddingType);
      }
      return [...current, embeddingType];
    });
  };
  const assetResults =
    response.assets.length > 0 ? response.assets : response.results;

  return (
    <main className="app-shell">
      <ProductTopbar
        active="search"
        connection={viewMode}
        status="Workspace Demo"
        workspaceControl={
          <WorkspaceSelect
            workspaceId={workspaceId}
            workspaces={workspaces}
            loading={workspacesLoading}
            onChange={handleWorkspaceChange}
          />
        }
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
              <p>默认检索原始内容，也可以按需组合主体、场景、风格等多个维度。</p>
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

              <fieldset className="asset-filters target-asset-types">
                <legend>目标素材类型</legend>
                {(Object.keys(ASSET_LABELS) as AssetType[]).map((assetType) => (
                  <label key={assetType}>
                    <input
                      type="checkbox"
                      name="target_asset_types"
                      value={assetType}
                      checked={assetTypes.includes(assetType)}
                      disabled={
                        assetTypes.includes(assetType) && assetTypes.length === 1
                      }
                      onChange={() => toggleAssetType(assetType)}
                    />
                    <span aria-hidden="true" />
                    {ASSET_LABELS[assetType]}
                  </label>
                ))}
                <p>维度选项会随目标素材类型变化。</p>
              </fieldset>

              <details className="dimension-select">
                <summary>
                  <span>
                    <small>检索维度</small>
                    <strong>
                      {embeddingTypes.length === 1
                        ? CHANNEL_LABELS[embeddingTypes[0]]
                        : `已选择 ${embeddingTypes.length} 个维度`}
                    </strong>
                  </span>
                  <b aria-hidden="true">⌄</b>
                </summary>
                <div className="dimension-options">
                  {SEARCH_DIMENSIONS.map((dimension) => {
                    const checked = embeddingTypes.includes(dimension.value);
                    const supportCount = dimensionSupportCount(
                      dimension.value,
                      assetTypes,
                    );
                    const available = supportCount > 0;
                    const partiallyAvailable =
                      available && supportCount < assetTypes.length;
                    return (
                      <label
                        key={dimension.value}
                        className={!available ? "dimension-unavailable" : undefined}
                      >
                        <input
                          type="checkbox"
                          name="embedding_types"
                          value={dimension.value}
                          checked={checked}
                          disabled={
                            !available || (checked && embeddingTypes.length === 1)
                          }
                          onChange={() => toggleEmbeddingType(dimension.value)}
                        />
                        <span aria-hidden="true" />
                        <em>{dimension.label}</em>
                        <code>
                          {!available
                            ? "当前类型不可用"
                            : partiallyAvailable
                              ? "仅图片 / 视频"
                              : dimension.value}
                        </code>
                      </label>
                    );
                  })}
                </div>
                <p>
                  文本多维检索会按已选维度生成针对性 Query；文本中的倾向可影响权重。
                  图片 / 视频内容不参与权重解析，source 由后端根据输入类型决定。
                </p>
              </details>

              <div className="search-options">
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
                  disabled={embeddingTypes.length === 1}
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
                    value={modelName}
                    onChange={(event) => setModelName(event.target.value)}
                    placeholder="Embedding 模型名称"
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
                      setResponse(EMPTY_RESPONSE);
                      setViewMode("live");
                      setError(null);
                    }}
                  >
                    清空结果
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
                  readOnly
                  aria-label="当前 Workspace ID"
                />
              </label>
              <label>
                User ID
                <input
                  value={createdBy}
                  onChange={(event) => setCreatedBy(event.target.value)}
                />
              </label>
            </details>
          </aside>

          <section className="results-panel">
            <div className="results-header">
              <div>
                <span className="eyebrow">RESULTS / 02</span>
                <h2>
                  {response.asset_total + response.cluster_total}
                  <small> 个结果，分为素材与聚类簇</small>
                </h2>
              </div>
              <div className="results-meta">
                <span>
                  {Math.round(response.timings.total_ms)} ms
                </span>
                <span>
                  Query 增强 {Math.round(response.timings.query_enhancement_ms)} ms
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

            <div className="result-set-tabs" role="tablist" aria-label="结果类型">
              <button
                className={resultSet === "assets" ? "active" : ""}
                type="button"
                role="tab"
                aria-selected={resultSet === "assets"}
                onClick={() => setResultSet("assets")}
              >
                <span>图片 / 视频</span>
                <strong>{response.asset_total}</strong>
              </button>
              <button
                className={resultSet === "clusters" ? "active" : ""}
                type="button"
                role="tab"
                aria-selected={resultSet === "clusters"}
                onClick={() => setResultSet("clusters")}
              >
                <span>相关聚类簇</span>
                <strong>{response.cluster_total}</strong>
              </button>
            </div>

            {response.degraded && (
              <div className="degraded-banner">
                <strong>部分链路已降级</strong>
                <span>{response.degraded_reasons.join("；")}</span>
              </div>
            )}

            {resultSet === "assets" && assetResults.length > 0 ? (
              <div className="results-grid">
                {assetResults.map((result, index) => (
                  <SearchResultCard
                    result={result}
                    index={index}
                    fusionMethod={response.fusion_method}
                    workspaceId={workspaceId}
                    key={result.asset_id}
                  />
                ))}
              </div>
            ) : resultSet === "clusters" && response.clusters.length > 0 ? (
              <div className="cluster-search-grid">
                {response.clusters.map((cluster, index) => (
                  <ClusterResultCard
                    cluster={cluster}
                    index={index}
                    key={cluster.cluster_capsule_id}
                  />
                ))}
              </div>
            ) : (
              <div className="empty-state">
                <span>0</span>
                <h2>
                  {resultSet === "clusters"
                    ? "命中素材暂未归入可用聚类簇"
                    : "没有找到相似素材"}
                </h2>
                <p>
                  {resultSet === "clusters"
                    ? "请先为相关 Feature 创建完成的 Cluster Run，或切换到素材结果。"
                    : "尝试减少过滤条件，或换一种更具体的场景描述。"}
                </p>
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
                    <strong>{capsule.query_text || "参考图片检索"}</strong>
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
                      <h2>
                        {selectedCapsule.query_text || "参考图片检索"}
                      </h2>
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
                          workspaceId={workspaceId}
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
