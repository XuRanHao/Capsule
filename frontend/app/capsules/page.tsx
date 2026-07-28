"use client";

import { useMemo, useState } from "react";
import DemoShell, { AssetThumb } from "../components/DemoShell";
import { DEMO_ASSETS, DEMO_CLUSTERS } from "../lib/demo-data";

type CapsuleKind = "cluster" | "search";
type CapsuleFilter = "all" | "recent" | "favorite";

type CapsuleRecord = {
  id: string;
  kind: CapsuleKind;
  name: string;
  summary: string;
  queryType?: string;
  query?: string;
  members: number;
  favorite: boolean;
  usedAt: string;
  timestamp: number;
  assetIds: string[];
  executions?: number;
};

const SEARCH_CAPSULES: CapsuleRecord[] = [
  {
    id: "search_capsule_twilight",
    kind: "search",
    name: "蓝紫色黄昏动画场景",
    summary: "文本精搜 · 6 路召回 · Weighted RRF · Seed 重排",
    queryType: "text",
    query: "蓝紫色黄昏动画场景，安静，有人物但不要文字水印",
    members: 20,
    favorite: true,
    usedAt: "刚刚",
    timestamp: 100,
    assetIds: ["asset_twilight_01", "asset_field_02", "asset_city_03"],
    executions: 4,
  },
  {
    id: "search_capsule_city",
    kind: "search",
    name: "城市边缘的小人物",
    summary: "图文精搜 · 保持构图 · 修改情绪",
    queryType: "image_text",
    query: "保持大面积天空和人物比例，更像蓝调时刻，排除高饱和霓虹",
    members: 16,
    favorite: false,
    usedAt: "今天 14:22",
    timestamp: 94,
    assetIds: ["asset_city_03", "asset_twilight_01"],
    executions: 2,
  },
  {
    id: "search_capsule_portrait",
    kind: "search",
    name: "金色轮廓光人像",
    summary: "图片快速模式 · Native Multimodal",
    queryType: "image",
    query: "参考图片",
    members: 12,
    favorite: false,
    usedAt: "昨天 21:08",
    timestamp: 72,
    assetIds: ["asset_reference_06"],
    executions: 1,
  },
  {
    id: "search_capsule_seaside",
    kind: "search",
    name: "海边奔跑与飞鸟",
    summary: "文本精搜 · Normalized Similarity",
    queryType: "text",
    query: "夕阳海边奔跑的人，飞鸟，柔和速度感",
    members: 18,
    favorite: false,
    usedAt: "7 月 26 日",
    timestamp: 48,
    assetIds: ["asset_seaside_05", "asset_field_02"],
    executions: 3,
  },
];

const CLUSTER_CAPSULES: CapsuleRecord[] = DEMO_CLUSTERS.map(
  (cluster, index) => ({
    id: cluster.id,
    kind: "cluster",
    name: cluster.name,
    summary: cluster.description,
    members: cluster.members,
    favorite: cluster.favorite,
    usedAt: index < 2 ? "今天" : "昨天",
    timestamp: 90 - index * 10,
    assetIds: cluster.assetIds,
  }),
);

export default function CapsulesPage() {
  const [kind, setKind] = useState<CapsuleKind>("search");
  const [filter, setFilter] = useState<CapsuleFilter>("all");
  const [records, setRecords] = useState([
    ...SEARCH_CAPSULES,
    ...CLUSTER_CAPSULES,
  ]);
  const [selectedId, setSelectedId] = useState(SEARCH_CAPSULES[0].id);
  const [snapshotMode, setSnapshotMode] = useState<"saved" | "latest">(
    "saved",
  );
  const [refreshing, setRefreshing] = useState(false);

  const visible = useMemo(
    () =>
      records
        .filter((record) => record.kind === kind)
        .filter((record) => filter !== "favorite" || record.favorite)
        .sort((left, right) => right.timestamp - left.timestamp),
    [filter, kind, records],
  );
  const selected =
    records.find((record) => record.id === selectedId && record.kind === kind) ??
    visible[0];
  const assets =
    selected?.assetIds
      .map((id) => DEMO_ASSETS.find((asset) => asset.id === id))
      .filter((asset) => asset !== undefined) ?? [];

  const switchKind = (nextKind: CapsuleKind) => {
    setKind(nextKind);
    const first = records.find((record) => record.kind === nextKind);
    if (first) setSelectedId(first.id);
  };

  const toggleFavorite = (id: string) => {
    setRecords((current) =>
      current.map((record) =>
        record.id === id
          ? { ...record, favorite: !record.favorite }
          : record,
      ),
    );
  };

  const refresh = () => {
    setRefreshing(true);
    window.setTimeout(() => {
      setRefreshing(false);
      setSnapshotMode("latest");
    }, 1000);
  };

  return (
    <DemoShell
      active="capsules"
      eyebrow="CAPSULE LIBRARY / 44.7"
      title="把一次发现，变成可复用的入口。"
      description="统一管理 Cluster Capsule 与 Search Capsule，快速回到最近使用和收藏的语义资产。"
      actions={
        <button className="secondary-action">导出 Capsule 清单</button>
      }
    >
      <section className="capsule-dashboard">
        <header className="capsule-controls">
          <div className="capsule-kind-tabs">
            <button
              className={kind === "cluster" ? "active" : ""}
              onClick={() => switchKind("cluster")}
            >
              <span>Cluster Capsule</span>
              <b>{records.filter((item) => item.kind === "cluster").length}</b>
            </button>
            <button
              className={kind === "search" ? "active" : ""}
              onClick={() => switchKind("search")}
            >
              <span>Search Capsule</span>
              <b>{records.filter((item) => item.kind === "search").length}</b>
            </button>
          </div>
          <div className="capsule-filter-tabs">
            <button
              className={filter === "all" ? "active" : ""}
              onClick={() => setFilter("all")}
            >
              全部
            </button>
            <button
              className={filter === "recent" ? "active" : ""}
              onClick={() => setFilter("recent")}
            >
              最近使用
            </button>
            <button
              className={filter === "favorite" ? "active" : ""}
              onClick={() => setFilter("favorite")}
            >
              ★ 收藏
            </button>
          </div>
        </header>

        <div className="capsule-browser">
          <aside className="unified-capsule-list">
            <header>
              <span>{filter === "favorite" ? "FAVORITES" : "RECENTLY USED"}</span>
              <b>{visible.length}</b>
            </header>
            {visible.map((record, index) => (
              <article
                className={selected?.id === record.id ? "active" : ""}
                onClick={() => setSelectedId(record.id)}
                key={record.id}
              >
                <div className={`capsule-glyph capsule-glyph-${record.kind}`}>
                  {record.kind === "search" ? "⌕" : "◎"}
                </div>
                <div>
                  <small>
                    {record.kind.toUpperCase()} · {record.usedAt}
                  </small>
                  <strong>{record.name}</strong>
                  <span>{record.summary}</span>
                  <footer>
                    <span>{record.members} ASSETS</span>
                    {record.executions && (
                      <span>{record.executions} EXECUTIONS</span>
                    )}
                  </footer>
                </div>
                <button
                  onClick={(event) => {
                    event.stopPropagation();
                    toggleFavorite(record.id);
                  }}
                  aria-label={record.favorite ? "取消收藏" : "收藏"}
                >
                  {record.favorite ? "★" : "☆"}
                </button>
                <i>{String(index + 1).padStart(2, "0")}</i>
              </article>
            ))}
            {!visible.length && (
              <div className="capsule-list-empty">这个分类里还没有 Capsule</div>
            )}
          </aside>

          <section className="unified-capsule-detail">
            {selected ? (
              <>
                <header>
                  <div>
                    <small>
                      {selected.kind === "search"
                        ? "SEARCH CAPSULE"
                        : "CLUSTER CAPSULE"}
                    </small>
                    <h2>{selected.name}</h2>
                    <p>{selected.summary}</p>
                  </div>
                  <button
                    className={selected.favorite ? "favorite-active" : ""}
                    onClick={() => toggleFavorite(selected.id)}
                  >
                    {selected.favorite ? "★ 已收藏" : "☆ 收藏"}
                  </button>
                </header>

                {selected.kind === "search" ? (
                  <>
                    <div className="saved-query">
                      <span>QUERY / {selected.queryType}</span>
                      <strong>{selected.query}</strong>
                    </div>
                    <div className="snapshot-switcher">
                      <button
                        className={snapshotMode === "saved" ? "active" : ""}
                        onClick={() => setSnapshotMode("saved")}
                      >
                        保存快照
                        <small>创建时的固定结果</small>
                      </button>
                      <button
                        className={snapshotMode === "latest" ? "active" : ""}
                        onClick={() => setSnapshotMode("latest")}
                      >
                        最新资产
                        <small>按当前索引重新搜索</small>
                      </button>
                      <button
                        className="snapshot-refresh"
                        disabled={refreshing}
                        onClick={refresh}
                      >
                        {refreshing ? "刷新中…" : "使用最新资产刷新 ↗"}
                      </button>
                    </div>
                  </>
                ) : (
                  <div className="cluster-capsule-summary">
                    <span>
                      <small>MEMBERS</small>
                      <strong>{selected.members}</strong>
                    </span>
                    <span>
                      <small>EMBEDDING</small>
                      <strong>native_multimodal</strong>
                    </span>
                    <span>
                      <small>LAST RUN</small>
                      <strong>{selected.usedAt}</strong>
                    </span>
                  </div>
                )}

                <div className="capsule-asset-grid">
                  {assets.map((asset, index) => (
                    <article key={asset.id}>
                      <AssetThumb
                        preview={asset.preview}
                        name={asset.name}
                        type={asset.type}
                      />
                      <div>
                        <small>
                          {snapshotMode === "latest" ? "LATEST" : "SNAPSHOT"} /{" "}
                          {String(index + 1).padStart(2, "0")}
                        </small>
                        <strong>{asset.name}</strong>
                        <span>{asset.sourceFile}</span>
                      </div>
                    </article>
                  ))}
                </div>

                <footer className="capsule-provenance">
                  <span>CAPSULE ID</span>
                  <strong>{selected.id}</strong>
                  <span>LAST USED</span>
                  <strong>{selected.usedAt}</strong>
                  {selected.executions && (
                    <>
                      <span>EXECUTIONS</span>
                      <strong>{selected.executions}</strong>
                    </>
                  )}
                </footer>
              </>
            ) : (
              <div className="capsule-detail-empty">请选择一个 Capsule</div>
            )}
          </section>
        </div>
      </section>
    </DemoShell>
  );
}
