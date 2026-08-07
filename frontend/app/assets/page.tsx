"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import DemoShell, { AssetThumb, StatusBadge } from "../components/DemoShell";
import {
  type AssetRecord,
  apiFetch,
  loadAssets,
} from "../lib/api";
import { useWorkspaceSelection, WorkspaceSelect } from "../lib/workspaces";

type AssetFilter = AssetRecord["asset_type"] | "all";

const TYPE_LABELS: Record<AssetFilter, string> = {
  all: "全部",
  image: "图片",
  video_segment: "视频片段",
  markdown_block: "Markdown",
  text_block: "文字段落",
};

function featureValues(asset: AssetRecord) {
  return Object.values(asset.asset_features)
    .map((feature) =>
      typeof feature === "string" ? feature : feature?.value,
    )
    .filter((value): value is string => Boolean(value))
    .slice(0, 2);
}

export default function AssetsPage() {
  const {
    workspaceId,
    workspaces,
    ready: workspaceReady,
    loading: workspacesLoading,
    setWorkspaceId,
    refreshWorkspaces,
  } = useWorkspaceSelection();
  const [view, setView] = useState<"grid" | "list">("grid");
  const [type, setType] = useState<AssetFilter>("all");
  const [status, setStatus] = useState("all");
  const [query, setQuery] = useState("");
  const [assets, setAssets] = useState<AssetRecord[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [clearNotice, setClearNotice] = useState<string | null>(null);
  const [clearing, setClearing] = useState(false);
  const loadRequestRef = useRef(0);

  const load = useCallback(async () => {
    if (!workspaceReady) return;
    const requestId = ++loadRequestRef.current;
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({
        workspace_id: workspaceId,
        limit: "500",
      });
      if (type !== "all") params.set("asset_type", type);
      if (status !== "all") params.set("processing_status", status);
      if (query.trim()) params.set("query", query.trim());
      const result = await loadAssets(params);
      if (requestId !== loadRequestRef.current) return;
      setAssets(result.items);
      setTotal(result.total);
    } catch (requestError) {
      if (requestId !== loadRequestRef.current) return;
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Asset 加载失败",
      );
    } finally {
      if (requestId === loadRequestRef.current) setLoading(false);
    }
  }, [query, status, type, workspaceId, workspaceReady]);

  useEffect(() => {
    const timer = window.setTimeout(() => void load(), 180);
    return () => window.clearTimeout(timer);
  }, [load]);

  const counts = useMemo(
    () =>
      assets.reduce<Record<string, number>>((result, asset) => {
        result[asset.asset_type] = (result[asset.asset_type] ?? 0) + 1;
        return result;
      }, {}),
    [assets],
  );

  const deleteWorkspace = async () => {
    const confirmed = window.confirm(
      `确定删除当前工作空间「${workspaceId}」吗？\n\n将永久删除该工作空间的 Asset、来源文件、处理任务、向量和媒体文件。其他工作空间不会受影响，此操作不可恢复。`,
    );
    if (!confirmed) return;

    setClearing(true);
    setError(null);
    setClearNotice(null);
    try {
      const result = await apiFetch<{
        workspace_id: string;
        workspace_deleted: boolean;
        assets_deleted: number;
        source_files_deleted: number;
        embeddings_deleted: number;
        jobs_deleted: number;
        vectors_deleted: number;
        objects_deleted: number;
        staging_paths_deleted: number;
        cancelled_jobs: number;
        cleanup_warnings: string[];
      }>(
        `/api/v1/workspaces/${encodeURIComponent(workspaceId)}?confirmation=${encodeURIComponent(workspaceId)}`,
        { method: "DELETE" },
      );
      setAssets([]);
      setTotal(0);
      setClearNotice(
        `已删除工作空间 ${result.workspace_id}：${result.assets_deleted} 个 Asset、${result.source_files_deleted} 个来源文件、${result.jobs_deleted} 个任务、${result.vectors_deleted} 条向量与 ${result.objects_deleted} 个媒体文件。${result.cleanup_warnings.length ? ` ${result.cleanup_warnings.join("；")}` : ""}`,
      );
      await refreshWorkspaces();
    } catch (requestError) {
      setError(
        requestError instanceof Error ? requestError.message : "工作空间删除失败",
      );
    } finally {
      setClearing(false);
    }
  };

  return (
    <DemoShell
      active="assets"
      workspaceControl={
        <WorkspaceSelect
          workspaceId={workspaceId}
          workspaces={workspaces}
          loading={workspacesLoading}
          onChange={(nextWorkspaceId) => {
            loadRequestRef.current += 1;
            setAssets([]);
            setTotal(0);
            setLoading(true);
            setClearNotice(null);
            setWorkspaceId(nextWorkspaceId);
          }}
        />
      }
      eyebrow="ASSET LIBRARY / LIVE"
      title="所有素材，都有语义。"
      description="这里直接读取 PostgreSQL 中的真实 Asset，并通过后端安全地加载本地图片和视频预览。"
      actions={
        <>
          <Link className="primary-action button-link" href="/import">
            ＋ 导入素材
          </Link>
          <button className="danger-action" disabled={clearing} onClick={deleteWorkspace}>
            {clearing ? "正在删除…" : "删除当前工作空间"}
          </button>
        </>
      }
    >
      <section className="asset-toolbar">
        <label className="asset-search">
          <span>⌕</span>
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="搜索文件名、Asset 名称或描述"
          />
          <small>LIVE</small>
        </label>
        <div className="view-toggle">
          <button
            className={view === "grid" ? "active" : ""}
            onClick={() => setView("grid")}
            aria-label="网格视图"
          >
            ⠿
          </button>
          <button
            className={view === "list" ? "active" : ""}
            onClick={() => setView("list")}
            aria-label="列表视图"
          >
            ☷
          </button>
        </div>
      </section>

      <section className="asset-filter-bar">
        <div className="type-tabs">
          {(Object.keys(TYPE_LABELS) as AssetFilter[]).map((item) => (
            <button
              className={type === item ? "active" : ""}
              onClick={() => setType(item)}
              key={item}
            >
              {TYPE_LABELS[item]}
              <small>{item === "all" ? total : counts[item] ?? 0}</small>
            </button>
          ))}
        </div>
        <div className="select-filters">
          <label>
            处理状态
            <select
              value={status}
              onChange={(event) => setStatus(event.target.value)}
            >
              <option value="all">全部状态</option>
              <option value="pending">等待中</option>
              <option value="processing">处理中</option>
              <option value="completed">已完成</option>
              <option value="partial_failed">部分失败</option>
              <option value="failed">失败</option>
            </select>
          </label>
          <button className="secondary-action" onClick={() => void load()}>
            刷新
          </button>
        </div>
      </section>

      <div className="asset-result-summary">
        <span>
          SHOWING <strong>{assets.length}</strong> OF {total}
        </span>
        <span>WORKSPACE · {workspaceId}</span>
      </div>

      {error && (
        <div className="asset-empty">
          <strong>无法读取 Asset</strong>
          <span>{error}</span>
          <button className="secondary-action" onClick={() => void load()}>
            重新加载
          </button>
        </div>
      )}
      {clearNotice && (
        <div className="asset-cleared-notice">
          <strong>工作空间已删除</strong>
          <span>{clearNotice}</span>
        </div>
      )}
      {!error && loading && (
        <div className="asset-empty">
          <strong>正在读取真实 Asset…</strong>
          <span>数据来自 PostgreSQL，不再使用演示素材。</span>
        </div>
      )}
      {!error && !loading && (
        <section className={`asset-library asset-library-${view}`}>
          {assets.map((asset, index) => (
            <Link
              className="library-asset-card"
              href={`/assets/${asset.asset_id}`}
              key={asset.asset_id}
            >
              <AssetThumb
                preview={asset.preview_url}
                name={asset.asset_name || asset.file_name}
                type={asset.asset_type}
              />
              <div className="library-asset-content">
                <header>
                  <span>{String(index + 1).padStart(2, "0")}</span>
                  <StatusBadge status={asset.processing_status} />
                </header>
                <h2>{asset.asset_name || asset.file_name}</h2>
                <p>
                  {asset.asset_description ||
                    "已完成文件解析，正在等待语义理解与 Embedding。"}
                </p>
                <div className="asset-feature-peek">
                  {featureValues(asset).map((value) => (
                    <span key={value}>{value}</span>
                  ))}
                  {!featureValues(asset).length && (
                    <span>{asset.embeddings.length} 个 Embedding 通道</span>
                  )}
                </div>
                <footer>
                  <div>
                    <strong>{asset.source_file.original_file_name}</strong>
                    <span>{asset.source_file.relative_path}</span>
                  </div>
                  <b>↗</b>
                </footer>
              </div>
            </Link>
          ))}
          {!assets.length && (
            <div className="asset-empty">
              <strong>当前筛选没有 Asset</strong>
              <span>导入素材或清空筛选条件后再试。</span>
            </div>
          )}
        </section>
      )}
    </DemoShell>
  );
}
