"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import DemoShell, {
  AssetThumb,
  StatusBadge,
} from "../components/DemoShell";
import {
  DEMO_ASSETS,
  type DemoAssetType,
} from "../lib/demo-data";

const TYPE_LABELS: Record<DemoAssetType | "all", string> = {
  all: "全部",
  image: "图片",
  video_segment: "视频片段",
  markdown_block: "文字段落",
};

export default function AssetsPage() {
  const [view, setView] = useState<"grid" | "list">("grid");
  const [type, setType] = useState<DemoAssetType | "all">("all");
  const [fileType, setFileType] = useState("all");
  const [sourceFile, setSourceFile] = useState("all");
  const [status, setStatus] = useState("all");
  const [query, setQuery] = useState("");

  const visible = useMemo(
    () =>
      DEMO_ASSETS.filter((asset) => {
        if (type !== "all" && asset.type !== type) return false;
        if (fileType !== "all" && asset.fileType !== fileType) return false;
        if (sourceFile !== "all" && asset.sourceFile !== sourceFile) return false;
        if (status !== "all" && asset.status !== status) return false;
        const normalized = query.trim().toLowerCase();
        return (
          !normalized ||
          asset.name.toLowerCase().includes(normalized) ||
          asset.description.toLowerCase().includes(normalized)
        );
      }),
    [fileType, query, sourceFile, status, type],
  );

  return (
    <DemoShell
      active="assets"
      eyebrow="ASSET LIBRARY / 44.3"
      title="所有素材，都有语义。"
      description="浏览切分后的图片、视频片段和 Markdown Block，按来源、类型和处理状态快速定位。"
      actions={
        <Link className="primary-action button-link" href="/import">
          ＋ 导入素材
        </Link>
      }
    >
      <section className="asset-toolbar">
        <label className="asset-search">
          <span>⌕</span>
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="搜索 Asset 名称或描述"
          />
          <small>⌘ K</small>
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
          {(Object.keys(TYPE_LABELS) as Array<DemoAssetType | "all">).map(
            (item) => (
              <button
                className={type === item ? "active" : ""}
                onClick={() => setType(item)}
                key={item}
              >
                {TYPE_LABELS[item]}
                <small>
                  {item === "all"
                    ? DEMO_ASSETS.length
                    : DEMO_ASSETS.filter((asset) => asset.type === item).length}
                </small>
              </button>
            ),
          )}
        </div>
        <div className="select-filters">
          <label>
            文件类型
            <select
              value={fileType}
              onChange={(event) => setFileType(event.target.value)}
            >
              <option value="all">全部</option>
              <option value="markdown">Markdown</option>
              <option value="video">视频</option>
              <option value="image">图片</option>
            </select>
          </label>
          <label>
            Source File
            <select
              value={sourceFile}
              onChange={(event) => setSourceFile(event.target.value)}
            >
              <option value="all">全部来源</option>
              {[...new Set(DEMO_ASSETS.map((asset) => asset.sourceFile))].map(
                (item) => (
                  <option value={item} key={item}>
                    {item}
                  </option>
                ),
              )}
            </select>
          </label>
          <label>
            处理状态
            <select
              value={status}
              onChange={(event) => setStatus(event.target.value)}
            >
              <option value="all">全部状态</option>
              <option value="completed">已完成</option>
              <option value="processing">处理中</option>
              <option value="partial_failed">部分失败</option>
            </select>
          </label>
        </div>
      </section>

      <div className="asset-result-summary">
        <span>
          SHOWING <strong>{visible.length}</strong> OF {DEMO_ASSETS.length}
        </span>
        <span>UPDATED 16:49</span>
      </div>

      <section className={`asset-library asset-library-${view}`}>
        {visible.map((asset, index) => (
          <Link
            className="library-asset-card"
            href={`/assets/${asset.id}`}
            key={asset.id}
          >
            <AssetThumb
              preview={asset.preview}
              name={asset.name}
              type={asset.type}
            />
            <div className="library-asset-content">
              <header>
                <span>{String(index + 1).padStart(2, "0")}</span>
                <StatusBadge status={asset.status} />
              </header>
              <h2>{asset.name}</h2>
              <p>{asset.description}</p>
              <div className="asset-feature-peek">
                {asset.features.slice(0, 2).map((feature) => (
                  <span key={feature.key}>{feature.value}</span>
                ))}
              </div>
              <footer>
                <div>
                  <strong>{asset.sourceFile}</strong>
                  <span>{asset.locator}</span>
                </div>
                <b>↗</b>
              </footer>
            </div>
          </Link>
        ))}
        {!visible.length && (
          <div className="asset-empty">
            <strong>没有符合条件的 Asset</strong>
            <span>清空筛选或尝试其他关键词。</span>
          </div>
        )}
      </section>
    </DemoShell>
  );
}
