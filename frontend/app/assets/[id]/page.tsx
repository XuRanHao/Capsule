"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import DemoShell, {
  AssetThumb,
  StatusBadge,
} from "../../components/DemoShell";
import SegmentVideoPlayer from "../../components/SegmentVideoPlayer";
import {
  type AssetRecord,
  apiFetch,
} from "../../lib/api";
import { useWorkspaceSelection, WorkspaceSelect } from "../../lib/workspaces";

const FEATURE_LABELS: Record<string, string> = {
  subject_content: "主体内容",
  scene_theme: "场景主题",
  visual_presentation: "视觉表现",
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

  const salienceRank = (value: number | "high" | "medium" | "low") =>
    typeof value === "number"
      ? -value
      : ({ high: 0, medium: 1, low: 2 } as const)[value];
  const features = Object.entries(asset.asset_features)
    .filter(([key]) => key in FEATURE_LABELS)
    .map(([key, raw]) => {
      const feature = typeof raw === "string" ? { value: raw } : raw;
      const legacyDescriptions = feature?.value
        ? feature.value
            .replaceAll(";", "；")
            .split("；")
            .map((description) => description.trim())
            .filter(Boolean)
        : [];
      const items = feature?.items?.length
        ? feature.items.slice().sort(
            (left, right) =>
              salienceRank(left.salience) - salienceRank(right.salience),
          )
        : legacyDescriptions.map((description, index) => ({
            description,
            salience: index === 0 ? ("high" as const) : ("medium" as const),
            status: (feature?.status || "inferred") as
              | "observed"
              | "inferred"
              | "metadata"
              | "user_supplied",
            evidence: index === 0 ? feature?.evidence || [] : [],
            ocr_confidence: null,
          }));
      return {
        key,
        label: FEATURE_LABELS[key] || key,
        applicability:
          feature?.applicability || (items.length ? "applicable" : "unknown"),
        items,
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
    asset.asset_type === "video_segment" &&
    Boolean(asset.playback || asset.content_url);

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
            <SegmentVideoPlayer
              playback={asset.playback}
              legacyContentUrl={asset.content_url}
              posterUrl={asset.preview_url}
              fallbackMimeType={asset.file_type}
            />
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
          <span>{features.length} / 3 DIMENSIONS</span>
        </header>
        <div className="feature-table">
          <div className="feature-row feature-head">
            <span>维度</span>
            <span>描述条目</span>
            <span>适用性</span>
            <span>证据</span>
          </div>
          {features.map((feature) => (
            <div className="feature-row" key={feature.key}>
              <strong>{feature.label}</strong>
              <span className="feature-value-cell">
                {feature.items.map((item) => (
                  <span className="feature-item-line" key={`${item.salience}:${item.description}`}>
                    <em
                      className={
                        typeof item.salience === "number"
                          ? "salience-relative"
                          : `salience-${item.salience}`
                      }
                    >
                      {typeof item.salience === "number"
                        ? item.salience.toFixed(2)
                        : item.salience}
                    </em>
                    <b>
                      {item.subject ? `${item.subject} · ` : ""}
                      {item.description}
                    </b>
                    <small>{item.status}</small>
                  </span>
                ))}
                {!feature.items.length && <small>暂无可用描述</small>}
              </span>
              <StatusBadge status={feature.applicability} />
              <details>
                <summary>
                  {feature.items.reduce((count, item) => count + item.evidence.length, 0)} 条证据
                </summary>
                <ul>
                  {feature.items.flatMap((item) =>
                    item.evidence.map((evidence) => (
                      <li key={`${item.description}:${evidence}`}>
                        {evidence}
                        {item.ocr_confidence != null && (
                          <small> · OCR {item.ocr_confidence.toFixed(2)}</small>
                        )}
                      </li>
                    )),
                  )}
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
