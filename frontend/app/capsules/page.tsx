"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import DemoShell, { AssetThumb, StatusBadge } from "../components/DemoShell";
import {
  type AssetRecord,
  type ClusterCapsule,
  type ClusterRun,
  type SearchCapsule,
  CREATED_BY,
  apiFetch,
  loadAssets,
} from "../lib/api";
import {
  useWorkspaceSelection,
  WorkspaceSelect,
} from "../lib/workspaces";

type CapsuleKind = "cluster" | "search";
type CapsuleFilter = "all" | "favorite";

type LiveCapsule =
  | { kind: "search"; id: string; search: SearchCapsule }
  | { kind: "cluster"; id: string; cluster: ClusterCapsule };

function inWorkspace<T extends { workspace_id: string }>(
  items: T[],
  workspaceId: string,
) {
  return items.filter((item) => item.workspace_id === workspaceId);
}

export default function CapsulesPage() {
  const {
    workspaceId,
    workspaces,
    ready: workspaceReady,
    loading: workspacesLoading,
    error: workspaceError,
    setWorkspaceId,
  } = useWorkspaceSelection();
  const [kind, setKind] = useState<CapsuleKind>("search");
  const [filter, setFilter] = useState<CapsuleFilter>("all");
  const [records, setRecords] = useState<LiveCapsule[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [assets, setAssets] = useState<AssetRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const listRequestRef = useRef(0);
  const assetRequestRef = useRef(0);
  const favoriteRequestRef = useRef(0);
  const workspaceIdRef = useRef(workspaceId);

  useEffect(() => {
    workspaceIdRef.current = workspaceId;
  }, [workspaceId]);

  const changeWorkspace = useCallback(
    (nextWorkspaceId: string) => {
      workspaceIdRef.current = nextWorkspaceId;
      listRequestRef.current += 1;
      assetRequestRef.current += 1;
      favoriteRequestRef.current += 1;
      setRecords([]);
      setSelectedId("");
      setAssets([]);
      setError(null);
      setLoading(true);
      setWorkspaceId(nextWorkspaceId);
    },
    [setWorkspaceId],
  );

  const load = useCallback(async () => {
    if (!workspaceReady || !workspaceId) return;
    const requestId = ++listRequestRef.current;
    const requestWorkspaceId = workspaceId;
    setLoading(true);
    try {
      const [searchPayload, runPayload] = await Promise.all([
        apiFetch<{ items: SearchCapsule[] }>(
          `/api/v1/search-capsules?workspace_id=${encodeURIComponent(requestWorkspaceId)}&created_by=${encodeURIComponent(CREATED_BY)}`,
        ),
        apiFetch<{ items: ClusterRun[] }>(
          `/api/v1/cluster-runs?workspace_id=${encodeURIComponent(requestWorkspaceId)}&limit=100`,
        ),
      ]);
      if (
        requestId !== listRequestRef.current ||
        workspaceIdRef.current !== requestWorkspaceId
      ) {
        return;
      }
      const searchItems = inWorkspace(
        searchPayload.items,
        requestWorkspaceId,
      );
      const completedRuns = inWorkspace(
        runPayload.items,
        requestWorkspaceId,
      ).filter(
        (run) => run.status === "completed",
      );
      const clusterPayloads = await Promise.all(
        completedRuns.map((run) =>
          apiFetch<{ items: ClusterCapsule[] }>(
            `/api/v1/cluster-runs/${encodeURIComponent(run.cluster_run_id)}/capsules?workspace_id=${encodeURIComponent(requestWorkspaceId)}`,
          ),
        ),
      );
      if (
        requestId !== listRequestRef.current ||
        workspaceIdRef.current !== requestWorkspaceId
      ) {
        return;
      }
      const next: LiveCapsule[] = [
        ...searchItems.map(
          (search): LiveCapsule => ({
            kind: "search",
            id: search.capsule_id,
            search,
          }),
        ),
        ...clusterPayloads.flatMap((payload) =>
          inWorkspace(payload.items, requestWorkspaceId).map(
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
          : "",
      );
      setError(null);
    } catch (requestError) {
      if (
        requestId !== listRequestRef.current ||
        workspaceIdRef.current !== requestWorkspaceId
      ) {
        return;
      }
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Capsule 加载失败",
      );
    } finally {
      if (
        requestId === listRequestRef.current &&
        workspaceIdRef.current === requestWorkspaceId
      ) {
        setLoading(false);
      }
    }
  }, [workspaceId, workspaceReady]);

  useEffect(() => {
    listRequestRef.current += 1;
    assetRequestRef.current += 1;
    favoriteRequestRef.current += 1;
    const timer = window.setTimeout(() => {
      setRecords([]);
      setSelectedId("");
      setAssets([]);
      setError(null);
      if (workspaceReady) void load();
    }, 0);
    return () => window.clearTimeout(timer);
  }, [load, workspaceId, workspaceReady]);

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
    const requestId = ++assetRequestRef.current;
    const requestWorkspaceId = workspaceId;
    const timer = window.setTimeout(() => {
      setAssets([]);
      if (!selected || !workspaceReady || !requestWorkspaceId) {
        return;
      }
      const loadSelectedAssets = async () => {
        let assetIds: string[] = [];
        if (selected.kind === "cluster") {
          assetIds = selected.cluster.representative_asset_ids;
        } else {
          const detail = await apiFetch<{
            workspace_id: string;
            latest_snapshot: {
              results: Array<{ asset_id: string }>;
            };
          }>(
            `/api/v1/search-capsules/${encodeURIComponent(selected.search.capsule_id)}?workspace_id=${encodeURIComponent(requestWorkspaceId)}&created_by=${encodeURIComponent(CREATED_BY)}`,
          );
          if (
            detail.workspace_id !== requestWorkspaceId ||
            requestId !== assetRequestRef.current ||
            workspaceIdRef.current !== requestWorkspaceId
          ) {
            return;
          }
          assetIds = detail.latest_snapshot.results.map(
            (item) => item.asset_id,
          );
        }
        if (!assetIds.length) {
          if (
            requestId === assetRequestRef.current &&
            workspaceIdRef.current === requestWorkspaceId
          ) {
            setAssets([]);
          }
          return;
        }
        const params = new URLSearchParams({
          workspace_id: requestWorkspaceId,
          limit: "100",
        });
        assetIds.forEach((assetId) => params.append("asset_id", assetId));
        const payload = await loadAssets(params);
        if (
          requestId === assetRequestRef.current &&
          workspaceIdRef.current === requestWorkspaceId
        ) {
          setAssets(inWorkspace(payload.items, requestWorkspaceId));
        }
      };
      void loadSelectedAssets().catch(() => {
        if (
          requestId === assetRequestRef.current &&
          workspaceIdRef.current === requestWorkspaceId
        ) {
          setAssets([]);
        }
      });
    }, 0);
    return () => {
      window.clearTimeout(timer);
      assetRequestRef.current += 1;
    };
  }, [selected, workspaceId, workspaceReady]);

  const switchKind = (nextKind: CapsuleKind) => {
    setKind(nextKind);
    setSelectedId(records.find((record) => record.kind === nextKind)?.id || "");
  };

  const toggleFavorite = async () => {
    if (!selected || selected.kind !== "search") return;
    const requestWorkspaceId = workspaceId;
    const requestId = ++favoriteRequestRef.current;
    const updated = await apiFetch<SearchCapsule>(
      `/api/v1/search-capsules/${encodeURIComponent(selected.search.capsule_id)}?workspace_id=${encodeURIComponent(requestWorkspaceId)}&created_by=${encodeURIComponent(CREATED_BY)}`,
      {
        method: "PATCH",
        body: JSON.stringify({
          is_favorite: !selected.search.is_favorite,
        }),
      },
    );
    if (
      requestId !== favoriteRequestRef.current ||
      workspaceIdRef.current !== requestWorkspaceId ||
      updated.workspace_id !== requestWorkspaceId
    ) {
      return;
    }
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
      workspaceControl={
        <WorkspaceSelect
          workspaceId={workspaceId}
          workspaces={workspaces}
          loading={workspacesLoading}
          onChange={changeWorkspace}
        />
      }
      eyebrow="CAPSULE LIBRARY / LIVE"
      title="把一次发现，变成可复用的入口。"
      description="统一读取真实 Cluster Capsule 和 Search Capsule，并展示其关联 Asset。"
      actions={
        <button
          className="secondary-action"
          disabled={!workspaceReady}
          onClick={() => void load()}
        >
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

        {(workspaceError || error) && (
          <div className="asset-empty">
            <strong>无法加载 Capsule</strong>
            <span>{workspaceError || error}</span>
          </div>
        )}
        {!workspaceError && !error && (!workspaceReady || loading) && (
          <div className="asset-empty">
            <strong>正在读取 Capsule…</strong>
          </div>
        )}
        {!workspaceError && !error && workspaceReady && !loading && (
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
