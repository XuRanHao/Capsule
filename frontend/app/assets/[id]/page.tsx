"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import DemoShell, {
  AssetThumb,
  StatusBadge,
} from "../../components/DemoShell";
import {
  type AssetRecord,
  apiFetch,
} from "../../lib/api";
import { useWorkspaceSelection, WorkspaceSelect } from "../../lib/workspaces";

const FEATURE_LABELS: Record<string, string> = {
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

export default function AssetDetailPage() {
  const {
    workspaceId,
    workspaces,
    ready: workspaceReady,
    loading: workspacesLoading,
    setWorkspaceId,
  } = useWorkspaceSelection();
  const pathname = usePathname();
  const assetId = pathname.split("/").filter(Boolean).at(-1);
  const [asset, setAsset] = useState<AssetRecord | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!assetId || !workspaceReady) return;
    void apiFetch<AssetRecord>(
      `/api/v1/assets/${encodeURIComponent(assetId)}?workspace_id=${encodeURIComponent(workspaceId)}`,
    )
      .then((loadedAsset) => {
        setAsset(loadedAsset);
        setError(null);
      })
      .catch((requestError: unknown) =>
        setError(
          requestError instanceof Error
            ? requestError.message
            : "Asset 加载失败",
        ),
      );
  }, [assetId, workspaceId, workspaceReady]);

  const workspaceControl = (
    <WorkspaceSelect
      workspaceId={workspaceId}
      workspaces={workspaces}
      loading={workspacesLoading}
      onChange={setWorkspaceId}
    />
  );

  if (!asset || asset.workspace_id !== workspaceId) {
    return (
      <DemoShell
        active="assets"
        workspaceControl={workspaceControl}
        eyebrow="ASSET DETAIL / LIVE"
        title={error ? "无法打开 Asset" : "正在读取 Asset…"}
        description={error || assetId || ""}
        actions={
          <Link className="secondary-action button-link" href="/assets">
            ← 返回列表
          </Link>
        }
      >
        <div className="asset-empty">
          <strong>{error || "正在读取 PostgreSQL 中的真实记录"}</strong>
        </div>
      </DemoShell>
    );
  }

  const features = Object.entries(asset.asset_features).map(([key, raw]) => {
    const feature =
      typeof raw === "string"
        ? { value: raw, status: "observed", confidence: 1, evidence: [] }
        : raw;
    return {
      key,
      label: FEATURE_LABELS[key] || key,
      value: feature.value || "暂无",
      status: feature.status || "unknown",
      confidence: feature.confidence ?? 0,
      evidence: feature.evidence || [],
      description: feature.description || null,
      sourcePath: feature.source_path || null,
    };
  });
  const locator = Object.entries(asset.source_locator)
    .map(([key, value]) => `${key}=${String(value)}`)
    .join(" · ");
  const context = asset.source_contexts
    .map((item) => item.text)
    .filter(Boolean)
    .join("\n");
  const playableVideo =
    asset.asset_type === "video_segment" && Boolean(asset.content_url);

  return (
    <DemoShell
      active="assets"
      workspaceControl={workspaceControl}
      eyebrow="ASSET DETAIL / LIVE"
      title={asset.asset_name || asset.file_name}
      description={`${asset.asset_id} · ${locator || "whole_file"}`}
      actions={
        <Link className="secondary-action button-link" href="/assets">
          ← 返回列表
        </Link>
      }
    >
      <div className="asset-detail-hero">
        <div className="asset-detail-preview">
          {playableVideo ? (
            <video
              className="asset-video-player"
              controls
              preload="metadata"
              poster={asset.preview_url ?? undefined}
            >
              <source src={asset.content_url ?? undefined} type="video/mp4" />
              当前浏览器不支持播放此视频片段。
            </video>
          ) : (
            <AssetThumb
              preview={asset.preview_url}
              name={asset.asset_name || asset.file_name}
              type={asset.asset_type}
            />
          )}
          <div className="preview-meta">
            <StatusBadge status={asset.processing_status} />
            <span>{asset.file_info.width ? `${asset.file_info.width} × ${asset.file_info.height}` : asset.file_type}</span>
          </div>
        </div>
        <section className="asset-core-info">
          <span className="eyebrow">SEMANTIC IDENTITY</span>
          <label>
            Asset Name
            <strong>{asset.asset_name || asset.file_name}</strong>
          </label>
          <label>
            Asset Description
            <p>
              {asset.asset_description ||
                "语义理解尚未完成；原始文件已经可用并可在列表中预览。"}
            </p>
          </label>
          {asset.error_message && (
            <div className="unsaved-banner">
              <span>{asset.error_message}</span>
            </div>
          )}
        </section>
      </div>

      <div className="asset-detail-grid">
        <section className="source-inspector">
          <header>
            <span className="eyebrow">SOURCE</span>
            <h2>来源与原始位置</h2>
          </header>
          <dl>
            <div>
              <dt>Source File</dt>
              <dd>{asset.source_file.original_file_name}</dd>
            </div>
            <div>
              <dt>相对路径</dt>
              <dd>{asset.source_file.relative_path}</dd>
            </div>
            <div>
              <dt>文件大小</dt>
              <dd>{(asset.source_file.file_size_bytes / 1024).toFixed(1)} KB</dd>
            </div>
            <div>
              <dt>原始位置</dt>
              <dd>{locator || "whole_file"}</dd>
            </div>
          </dl>
          {context && (
            <blockquote>
              <span>关联段落</span>
              <p>{context}</p>
            </blockquote>
          )}
        </section>

        <section className="embedding-inspector">
          <header>
            <span className="eyebrow">EMBEDDING GROUP</span>
            <h2>向量状态</h2>
          </header>
          {asset.embeddings.map((embedding) => (
            <div key={embedding.embedding_type}>
              <span>
                <i className={`embedding-dot ${embedding.status}`} />
                {embedding.embedding_type}
              </span>
              <StatusBadge status={embedding.status} />
              <small>{embedding.model_name}</small>
            </div>
          ))}
          {!asset.embeddings.length && (
            <div>
              <span>尚无 Embedding</span>
              <StatusBadge status="pending" />
              <small>REV {asset.embedding_revision}</small>
            </div>
          )}
        </section>
      </div>

      <section className="feature-editor">
        <header>
          <div>
            <span className="eyebrow">ASSET FEATURES</span>
            <h2>多维语义特征</h2>
          </div>
          <span>{features.length} / 10 DIMENSIONS</span>
        </header>
        <div className="feature-table">
          <div className="feature-row feature-head">
            <span>维度</span>
            <span>Effective Value</span>
            <span>状态</span>
            <span>Confidence</span>
            <span>Evidence</span>
          </div>
          {features.map((feature) => (
            <div className="feature-row" key={feature.key}>
              <strong>{feature.label}</strong>
              <span className="feature-value-cell">
                <b>{feature.value}</b>
                {feature.description && <small>{feature.description}</small>}
                {feature.sourcePath && (
                  <code title={feature.sourcePath}>{feature.sourcePath}</code>
                )}
              </span>
              <StatusBadge status={feature.status} />
              <span className="confidence-cell">
                <i style={{ width: `${feature.confidence * 100}%` }} />
                <b>{feature.confidence.toFixed(2)}</b>
              </span>
              <details>
                <summary>{feature.evidence.length} 条证据</summary>
                <ul>
                  {feature.evidence.map((evidence) => (
                    <li key={evidence}>{evidence}</li>
                  ))}
                </ul>
              </details>
            </div>
          ))}
          {!features.length && (
            <div className="asset-empty">
              <strong>语义理解正在排队</strong>
              <span>完成后这里会显示 10 个真实 Feature。</span>
            </div>
          )}
        </div>
      </section>
    </DemoShell>
  );
}
