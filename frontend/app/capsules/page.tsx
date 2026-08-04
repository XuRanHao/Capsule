"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import DemoShell, { AssetThumb, StatusBadge } from "../components/DemoShell";
import {
  type AssetRecord,
  type ClusterCapsule,
  type ClusterRun,
  type SearchCapsule,
  CREATED_BY,
  WORKSPACE_ID,
  apiFetch,
  loadAssets,
} from "../lib/api";

type CapsuleKind = "cluster" | "search";
type CapsuleFilter = "all" | "favorite";

type LiveCapsule =
  | { kind: "search"; id: string; search: SearchCapsule }
  | { kind: "cluster"; id: string; cluster: ClusterCapsule };

export default function CapsulesPage() {
  const [kind, setKind] = useState<CapsuleKind>("search");
  const [filter, setFilter] = useState<CapsuleFilter>("all");
  const [records, setRecords] = useState<LiveCapsule[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [assets, setAssets] = useState<AssetRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [searchPayload, runPayload] = await Promise.all([
        apiFetch<{ items: SearchCapsule[] }>(
          `/api/v1/search-capsules?workspace_id=${WORKSPACE_ID}&created_by=${CREATED_BY}`,
        ),
        apiFetch<{ items: ClusterRun[] }>(
          `/api/v1/cluster-runs?workspace_id=${WORKSPACE_ID}&limit=100`,
        ),
      ]);
      const completedRuns = runPayload.items.filter(
        (run) => run.status === "completed",
      );
      const clusterPayloads = await Promise.all(
        completedRuns.map((run) =>
          apiFetch<{ items: ClusterCapsule[] }>(
            `/api/v1/cluster-runs/${run.cluster_run_id}/capsules?workspace_id=${WORKSPACE_ID}`,
          ),
        ),
      );
      const next: LiveCapsule[] = [
        ...searchPayload.items.map(
          (search): LiveCapsule => ({
            kind: "search",
            id: search.capsule_id,
            search,
          }),
        ),
        ...clusterPayloads.flatMap((payload) =>
          payload.items.map(
            (cluster): LiveCapsule => ({
              kind: "cluster",
              id: cluster.cluster_capsule_id,
              cluster,
            }),
          ),
        ),
      ];
      setRecords(next);
      setSelectedId((current) =>
        current && next.some((record) => record.id === current)
          ? current
          : next.find((record) => record.kind === kind)?.id || "",
      );
      setError(null);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Capsule 加载失败",
      );
    } finally {
      setLoading(false);
    }
  }, [kind]);

  useEffect(() => {
    const timer = window.setTimeout(() => void load(), 0);
    return () => window.clearTimeout(timer);
  }, [load]);

  const visible = useMemo(
    () =>
      records.filter((record) => {
        if (record.kind !== kind) return false;
        if (filter !== "favorite") return true;
        return record.kind === "search"
          ? record.search.is_favorite
          : record.cluster.is_favorite;
      }),
    [filter, kind, records],
  );
  const selected =
    records.find((record) => record.id === selectedId && record.kind === kind) ??
    visible[0];

  useEffect(() => {
    const timer = window.setTimeout(() => {
      if (!selected) {
        setAssets([]);
        return;
      }
      const loadSelectedAssets = async () => {
        let assetIds: string[] = [];
        if (selected.kind === "cluster") {
          assetIds = selected.cluster.representative_asset_ids;
        } else {
          const detail = await apiFetch<{
            latest_snapshot: {
              results: Array<{ asset_id: string }>;
            };
          }>(
            `/api/v1/search-capsules/${selected.search.capsule_id}?workspace_id=${WORKSPACE_ID}&created_by=${CREATED_BY}`,
          );
          assetIds = detail.latest_snapshot.results.map(
            (item) => item.asset_id,
          );
        }
        if (!assetIds.length) {
          setAssets([]);
          return;
        }
        const params = new URLSearchParams({
          workspace_id: WORKSPACE_ID,
          limit: "100",
        });
        assetIds.forEach((assetId) => params.append("asset_id", assetId));
        setAssets((await loadAssets(params)).items);
      };
      void loadSelectedAssets().catch(() => setAssets([]));
    }, 0);
    return () => window.clearTimeout(timer);
  }, [selected]);

  const switchKind = (nextKind: CapsuleKind) => {
    setKind(nextKind);
    setSelectedId(records.find((record) => record.kind === nextKind)?.id || "");
  };

  const toggleFavorite = async () => {
    if (!selected || selected.kind !== "search") return;
    await apiFetch(
      `/api/v1/search-capsules/${selected.search.capsule_id}?workspace_id=${WORKSPACE_ID}&created_by=${CREATED_BY}`,
      {
        method: "PATCH",
        body: JSON.stringify({
          is_favorite: !selected.search.is_favorite,
        }),
      },
    );
    await load();
  };

  const name = selected
    ? selected.kind === "search"
      ? selected.search.query_text || "参考图片检索"
      : selected.cluster.effective_name
    : "";
  const summary = selected
    ? selected.kind === "search"
      ? selected.search.query_text || "参考图片检索"
      : selected.cluster.effective_description
    : "";

  return (
    <DemoShell
      active="capsules"
      eyebrow="CAPSULE LIBRARY / LIVE"
      title="把一次发现，变成可复用的入口。"
      description="统一读取真实 Cluster Capsule 和 Search Capsule，并展示其关联 Asset。"
      actions={
        <button className="secondary-action" onClick={() => void load()}>
          刷新
        </button>
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
              className={filter === "favorite" ? "active" : ""}
              onClick={() => setFilter("favorite")}
            >
              已收藏
            </button>
          </div>
        </header>

        {error && (
          <div className="asset-empty">
            <strong>无法加载 Capsule</strong>
            <span>{error}</span>
          </div>
        )}
        {!error && loading && (
          <div className="asset-empty">
            <strong>正在读取 Capsule…</strong>
          </div>
        )}
        {!error && !loading && (
          <div className="capsule-layout">
            <aside className="capsule-list">
              {visible.map((record) => {
                const recordName =
                  record.kind === "search"
                    ? record.search.query_text || "参考图片检索"
                    : record.cluster.effective_name;
                const count =
                  record.kind === "search"
                    ? record.search.result_count
                    : record.cluster.member_count;
                const favorite =
                  record.kind === "search"
                    ? record.search.is_favorite
                    : record.cluster.is_favorite;
                return (
                  <button
                    className={selected?.id === record.id ? "active" : ""}
                    onClick={() => setSelectedId(record.id)}
                    key={record.id}
                  >
                    <span>{record.kind.toUpperCase()}</span>
                    <strong>{recordName}</strong>
                    <small>{count} ASSETS</small>
                    <b>{favorite ? "★" : "☆"}</b>
                  </button>
                );
              })}
              {!visible.length && (
                <div className="asset-empty">
                  <strong>暂无真实 Capsule</strong>
                  <span>
                    {kind === "cluster"
                      ? "先在 Cluster 页面创建 Run。"
                      : "在搜索时选择保存 Capsule。"}
                  </span>
                </div>
              )}
            </aside>

            <section className="capsule-detail">
              {selected ? (
                <>
                  <header>
                    <div>
                      <span className="eyebrow">
                        {selected.kind.toUpperCase()} CAPSULE
                      </span>
                      <h2>{name}</h2>
                      <p>{summary}</p>
                    </div>
                    <div>
                      <StatusBadge status="completed" />
                      {selected.kind === "search" && (
                        <button onClick={() => void toggleFavorite()}>
                          {selected.search.is_favorite ? "★ 已收藏" : "☆ 收藏"}
                        </button>
                      )}
                    </div>
                  </header>
                  <div className="representative-strip">
                    {assets.slice(0, 8).map((asset, index) => (
                      <article key={asset.asset_id}>
                        <AssetThumb
                          preview={asset.preview_url}
                          name={asset.asset_name || asset.file_name}
                          type={asset.asset_type}
                        />
                        <span>ASSET {index + 1}</span>
                        <strong>{asset.asset_name || asset.file_name}</strong>
                      </article>
                    ))}
                  </div>
                </>
              ) : (
                <div className="asset-empty">
                  <strong>选择一个 Capsule 查看详情</strong>
                </div>
              )}
            </section>
          </div>
        )}
      </section>
    </DemoShell>
  );
}
