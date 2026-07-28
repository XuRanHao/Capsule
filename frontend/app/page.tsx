"use client";

import {
  FormEvent,
  KeyboardEvent,
  useMemo,
  useState,
} from "react";

type QueryType = "text" | "image" | "image_text";
type AssetType = "image" | "video_segment" | "markdown_block";

type MatchedChannel = {
  channel: string;
  embedding_type: string;
  rank: number;
  similarity: number;
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
  };
  score: number;
  matched_channels: MatchedChannel[];
  matched_feature: string | null;
};

type SearchResponse = {
  query: {
    query_type: QueryType;
    query_text: string | null;
    query_image_url: string | null;
  };
  total: number;
  degraded: boolean;
  degraded_reasons: string[];
  results: SearchResult[];
};

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

const QUERY_TYPES: Array<{
  value: QueryType;
  label: string;
  marker: string;
}> = [
  { value: "text", label: "文字", marker: "T" },
  { value: "image", label: "图片", marker: "I" },
  { value: "image_text", label: "图文", marker: "T＋I" },
];

const CHANNEL_LABELS: Record<string, string> = {
  native_multimodal: "原生多模态",
  "native_multimodal:image": "图片原生",
  "native_multimodal:text": "文字原生",
  asset_description: "内容描述",
  subject_content: "主体内容",
  visual_style: "视觉风格",
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
        rrf_contribution: 0.01639,
      },
      {
        channel: "visual_style",
        embedding_type: "visual_style",
        rank: 2,
        similarity: 0.901,
        rrf_contribution: 0.00968,
      },
      {
        channel: "asset_description",
        embedding_type: "asset_description",
        rank: 4,
        similarity: 0.884,
        rrf_contribution: 0.0125,
      },
    ],
    matched_feature: "日系动画电影感，柔和颗粒",
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
        text_block_index: 21,
      },
    ],
    source_locator: {
      start_time_seconds: 42.3,
      end_time_seconds: 49.8,
    },
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
        rrf_contribution: 0.01311,
      },
      {
        channel: "native_multimodal",
        embedding_type: "native_multimodal",
        rank: 5,
        similarity: 0.867,
        rrf_contribution: 0.01538,
      },
    ],
    matched_feature: "人物、马匹、金色田野",
  },
  {
    asset_id: "asset_demo_city_03",
    asset_type: "image",
    asset_name: "城市最后一束光",
    asset_description:
      "高架桥上的人物望向被夕光覆盖的城市，宽银幕构图和大面积天空强化了孤独感。",
    asset_features: {},
    source_contexts: [
      {
        text: "城市部分需要留出足够的天空，不要拥挤，人物应该只是画面中的小锚点。",
        relation_type: "preceding_text",
        text_block_index: 31,
      },
    ],
    source_locator: { block_index: 32 },
    preview_uri:
      "https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?auto=format&fit=crop&w=1200&q=82",
    source_file: {
      source_file_id: "src_demo_moodboard",
      original_file_name: "2026-夏日情绪板.md",
      file_type: "markdown",
      relative_path: "灵感库/视觉参考/2026-夏日情绪板.md",
    },
    score: 0.03492,
    matched_channels: [
      {
        channel: "visual_style",
        embedding_type: "visual_style",
        rank: 1,
        similarity: 0.912,
        rrf_contribution: 0.00984,
      },
      {
        channel: "asset_description",
        embedding_type: "asset_description",
        rank: 7,
        similarity: 0.821,
        rrf_contribution: 0.01194,
      },
    ],
    matched_feature: "宽银幕、逆光、城市远景",
  },
  {
    asset_id: "asset_demo_notes_04",
    asset_type: "markdown_block",
    asset_name: "黄昏氛围关键词",
    asset_description:
      "关于暖色暮光、蓝调时刻、长阴影和克制叙事的创作笔记。",
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
        rrf_contribution: 0.0129,
      },
    ],
    matched_feature: null,
  },
];

function secondsToTime(value: unknown) {
  if (typeof value !== "number") return null;
  const minutes = Math.floor(value / 60);
  const seconds = Math.floor(value % 60)
    .toString()
    .padStart(2, "0");
  return `${minutes}:${seconds}`;
}

function getPreviewUrl(uri: string | null) {
  if (!uri) return null;
  return /^(https?:|data:image\/)/.test(uri) ? uri : null;
}

function SearchResultCard({
  result,
  index,
}: {
  result: SearchResult;
  index: number;
}) {
  const [imageFailed, setImageFailed] = useState(false);
  const previewUrl = getPreviewUrl(result.preview_uri);
  const startTime = secondsToTime(result.source_locator.start_time_seconds);
  const context = result.source_contexts.find((item) => item.text)?.text;

  return (
    <article className={`result-card result-card-${index % 4}`}>
      <div className="result-visual">
        {previewUrl && !imageFailed ? (
          // Asset previews can come from arbitrary user-owned object storage.
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
          <small>RRF</small>
          <strong>{result.score.toFixed(4)}</strong>
        </div>
      </div>

      <div className="result-body">
        <div className="result-index">0{index + 1}</div>
        <h2>{result.asset_name || "未命名 Asset"}</h2>
        <p className="result-description">{result.asset_description}</p>

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
              title={`通道排名 ${channel.rank}，RRF 贡献 ${channel.rrf_contribution.toFixed(5)}`}
            >
              <span>
                {CHANNEL_LABELS[channel.channel] ?? channel.embedding_type}
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

        <footer className="source-footer">
          <div className="file-mark" aria-hidden="true">
            {result.source_file.file_type.slice(0, 2).toUpperCase()}
          </div>
          <div>
            <strong>{result.source_file.original_file_name}</strong>
            <span>{result.source_file.relative_path}</span>
          </div>
        </footer>
      </div>
    </article>
  );
}

export default function Home() {
  const [queryType, setQueryType] = useState<QueryType>("text");
  const [queryText, setQueryText] = useState("蓝紫色黄昏动画场景");
  const [queryImageUrl, setQueryImageUrl] = useState("");
  const [workspaceId, setWorkspaceId] = useState("workspace_demo");
  const [apiBaseUrl, setApiBaseUrl] = useState(API_BASE_URL);
  const [assetTypes, setAssetTypes] = useState<AssetType[]>([
    "image",
    "video_segment",
  ]);
  const [response, setResponse] = useState<SearchResponse>({
    query: {
      query_type: "text",
      query_text: "蓝紫色黄昏动画场景",
      query_image_url: null,
    },
    total: DEMO_RESULTS.length,
    degraded: false,
    degraded_reasons: [],
    results: DEMO_RESULTS,
  });
  const [viewMode, setViewMode] = useState<"demo" | "live">("demo");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [elapsedMs, setElapsedMs] = useState<number | null>(null);

  const imageQueryEnabled =
    queryType === "image" || queryType === "image_text";
  const textQueryEnabled = queryType === "text" || queryType === "image_text";

  const validationMessage = useMemo(() => {
    if (!workspaceId.trim()) return "请填写 Workspace ID";
    if (textQueryEnabled && !queryText.trim()) return "请输入检索文字";
    if (imageQueryEnabled && !queryImageUrl.trim()) return "请输入查询图片 URL";
    return null;
  }, [
    imageQueryEnabled,
    queryImageUrl,
    queryText,
    textQueryEnabled,
    workspaceId,
  ]);

  const toggleAssetType = (assetType: AssetType) => {
    setAssetTypes((current) =>
      current.includes(assetType)
        ? current.filter((item) => item !== assetType)
        : [...current, assetType],
    );
  };

  const submitSearch = async (event: FormEvent) => {
    event.preventDefault();
    if (validationMessage) {
      setError(validationMessage);
      return;
    }

    setLoading(true);
    setError(null);
    const started = performance.now();
    try {
      const apiResponse = await fetch(
        `${apiBaseUrl.replace(/\/$/, "")}/api/v1/search`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            workspace_id: workspaceId.trim(),
            query_type: queryType,
            query_text: textQueryEnabled ? queryText.trim() : null,
            query_image_url: imageQueryEnabled ? queryImageUrl.trim() : null,
            filters: { asset_type: assetTypes },
            top_k: 20,
          }),
        },
      );

      const payload = (await apiResponse.json()) as
        | SearchResponse
        | { detail?: { message?: string } | string };
      if (!apiResponse.ok) {
        const detail =
          "detail" in payload && typeof payload.detail === "object"
            ? payload.detail?.message
            : "detail" in payload
              ? payload.detail
              : null;
        throw new Error(detail || `搜索请求失败（${apiResponse.status}）`);
      }
      setResponse(payload as SearchResponse);
      setViewMode("live");
      setElapsedMs(Math.round(performance.now() - started));
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

  const handleQueryKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  };

  const restoreDemo = () => {
    setResponse({
      query: {
        query_type: "text",
        query_text: "蓝紫色黄昏动画场景",
        query_image_url: null,
      },
      total: DEMO_RESULTS.length,
      degraded: false,
      degraded_reasons: [],
      results: DEMO_RESULTS,
    });
    setViewMode("demo");
    setElapsedMs(null);
    setError(null);
  };

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Capsule 首页">
          <span className="brand-orbit" aria-hidden="true">
            <i />
          </span>
          <strong>CAPSULE</strong>
          <em>多模态记忆检索</em>
        </a>
        <div className="topbar-meta">
          <span className={`connection-dot ${viewMode}`} />
          <span>{viewMode === "live" ? "实时结果" : "演示数据"}</span>
          <b>{workspaceId || "未选择 Workspace"}</b>
        </div>
      </header>

      <div className="workspace" id="top">
        <aside className="query-panel">
          <div className="query-heading">
            <span className="eyebrow">SEARCH / 01</span>
            <h1>
              搜到你记得的
              <br />
              那一幕。
            </h1>
            <p>
              用一句话、一张图，或两者一起，从你的素材库里找回内容与上下文。
            </p>
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
                    value={item.value}
                    checked={queryType === item.value}
                    onChange={() => {
                      setQueryType(item.value);
                      setError(null);
                    }}
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
                  placeholder="例如：蓝紫色黄昏，人物站在城市边缘…"
                  rows={5}
                />
                <small>⌘ Enter 快速检索</small>
              </label>
            )}

            {imageQueryEnabled && (
              <label className="field-group">
                <span>查询图片 URL</span>
                <input
                  type="url"
                  value={queryImageUrl}
                  onChange={(event) => setQueryImageUrl(event.target.value)}
                  placeholder="https://…/reference.jpg"
                />
                <small>POC 阶段使用模型可访问的 HTTP 图片地址</small>
              </label>
            )}

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

            <button
              className="search-button"
              type="submit"
              disabled={loading}
            >
              <span>{loading ? "正在检索…" : "开始检索"}</span>
              <b aria-hidden="true">↗</b>
            </button>

            {error && (
              <div className="error-message" role="alert">
                <strong>检索未完成</strong>
                <span>{error}</span>
                {viewMode !== "demo" && (
                  <button type="button" onClick={restoreDemo}>
                    查看演示结果
                  </button>
                )}
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
                <small> 个相关片段</small>
              </h2>
            </div>
            <div className="results-meta">
              {elapsedMs !== null && <span>{elapsedMs} ms</span>}
              <span>Weighted RRF</span>
              <button type="button" onClick={restoreDemo}>
                演示数据
              </button>
            </div>
          </div>

          {response.degraded && (
            <div className="degraded-banner">
              <strong>部分通道已降级</strong>
              <span>
                {response.degraded_reasons.join("；") ||
                  "其余通道的结果仍然可用"}
              </span>
            </div>
          )}

          {response.results.length > 0 ? (
            <div className="results-grid">
              {response.results.map((result, index) => (
                <SearchResultCard
                  result={result}
                  index={index}
                  key={result.asset_id}
                />
              ))}
            </div>
          ) : (
            <div className="empty-state">
              <span>0</span>
              <h2>没有找到相似素材</h2>
              <p>尝试减少素材类型限制，或换一种更具体的场景描述。</p>
            </div>
          )}
        </section>
      </div>
    </main>
  );
}
