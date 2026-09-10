"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import DemoShell, { AssetThumb, StatusBadge } from "../components/DemoShell";
import {
  type AssetRecord,
  type SearchCapsule,
  CREATED_BY,
  apiFetch,
  loadAssets,
} from "../lib/api";
import { useWorkspaceSelection, WorkspaceSelect } from "../lib/workspaces";

type CapsuleFilter = "all" | "favorite";

function inWorkspace<T extends { workspace_id: string }>(items: T[], workspaceId: string) {
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
  const [filter, setFilter] = useState<CapsuleFilter>("all");
  const [records, setRecords] = useState<SearchCapsule[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [assets, setAssets] = useState<AssetRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const requestRef = useRef(0);

  const load = useCallback(async () => {
    if (!workspaceReady || !workspaceId) return;
    const requestId = ++requestRef.current;
    setLoading(true);
    try {
      const payload = await apiFetch<{ items: SearchCapsule[] }>(
        `/api/v1/search-capsules?workspace_id=${encodeURIComponent(workspaceId)}&created_by=${encodeURIComponent(CREATED_BY)}`,
      );
      if (requestId !== requestRef.current) return;
      const next = inWorkspace(payload.items, workspaceId);
      setRecords(next);
      setSelectedId((current) => current && next.some((item) => item.capsule_id === current) ? current : next[0]?.capsule_id ?? "");
      setError(null);
    } catch (requestError) {
      if (requestId !== requestRef.current) return;
      setError(requestError instanceof Error ? requestError.message : "Capsule 加载失败");
    } finally {
      if (requestId === requestRef.current) setLoading(false);
    }
  }, [workspaceId, workspaceReady]);

  useEffect(() => {
    setRecords([]);
    setSelectedId("");
    setAssets([]);
    if (workspaceReady) void load();
  }, [load, workspaceReady]);

  const visible = useMemo(
    () => records.filter((record) => filter === "all" || record.is_favorite),
    [filter, records],
  );
  const selected = visible.find((record) => record.capsule_id === selectedId) ?? visible[0];

  useEffect(() => {
    if (!selected || !workspaceReady || !workspaceId) {
      setAssets([]);
      return;
    }
    const requestId = ++requestRef.current;
    void apiFetch<{
      workspace_id: string;
      latest_snapshot: { results: Array<{ asset_id: string }> };
    }>(
      `/api/v1/search-capsules/${encodeURIComponent(selected.capsule_id)}?workspace_id=${encodeURIComponent(workspaceId)}&created_by=${encodeURIComponent(CREATED_BY)}`,
    )
      .then(async (detail) => {
        if (detail.workspace_id !== workspaceId || requestId !== requestRef.current) return;
        const params = new URLSearchParams({ workspace_id: workspaceId, limit: "100" });
        detail.latest_snapshot.results.forEach((item) => params.append("asset_id", item.asset_id));
        const payload = await loadAssets(params);
        if (requestId === requestRef.current) setAssets(inWorkspace(payload.items, workspaceId));
      })
      .catch(() => {
        if (requestId === requestRef.current) setAssets([]);
      });
  }, [selected, workspaceId, workspaceReady]);

  const toggleFavorite = async () => {
    if (!selected || !workspaceId) return;
    await apiFetch<SearchCapsule>(
      `/api/v1/search-capsules/${encodeURIComponent(selected.capsule_id)}?workspace_id=${encodeURIComponent(workspaceId)}&created_by=${encodeURIComponent(CREATED_BY)}`,
      { method: "PATCH", body: JSON.stringify({ is_favorite: !selected.is_favorite }) },
    );
    await load();
  };

  return (
    <DemoShell
      active="capsules"
      workspaceControl={<WorkspaceSelect workspaceId={workspaceId} workspaces={workspaces} loading={workspacesLoading} onChange={setWorkspaceId} />}
      eyebrow="CAPSULE LIBRARY / LIVE"
      title="把一次发现，变成可复用的入口。"
      description="统一读取搜索 Capsule，并展示其关联 Asset。"
      actions={<button className="secondary-action" disabled={!workspaceReady} onClick={() => void load()}>刷新</button>}
    >
      <section className="capsule-dashboard">
        <header className="capsule-controls">
          <div className="capsule-kind-tabs"><strong>Search Capsule</strong><b>{records.length}</b></div>
          <div className="capsule-filter-tabs">
            <button className={filter === "all" ? "active" : ""} onClick={() => setFilter("all")}>全部</button>
            <button className={filter === "favorite" ? "active" : ""} onClick={() => setFilter("favorite")}>已收藏</button>
          </div>
        </header>
        {(workspaceError || error) && <div className="asset-empty"><strong>无法加载 Capsule</strong><span>{workspaceError || error}</span></div>}
        {!workspaceError && !error && (!workspaceReady || loading) && <div className="asset-empty"><strong>正在读取 Capsule…</strong></div>}
        {!workspaceError && !error && workspaceReady && !loading && (
          <div className="capsule-layout">
            <aside className="capsule-list">
              {visible.map((record) => (
                <button className={selected?.capsule_id === record.capsule_id ? "active" : ""} onClick={() => setSelectedId(record.capsule_id)} key={record.capsule_id}>
                  <span>SEARCH</span><strong>{record.query_text || "参考图片检索"}</strong><small>{record.result_count} ASSETS</small><b>{record.is_favorite ? "★" : "☆"}</b>
                </button>
              ))}
              {!visible.length && <div className="asset-empty"><strong>暂无真实 Capsule</strong><span>在搜索时选择保存 Capsule。</span></div>}
            </aside>
            <section className="capsule-detail">
              {selected ? (
                <>
                  <header><div><span className="eyebrow">SEARCH CAPSULE</span><h2>{selected.query_text || "参考图片检索"}</h2><p>检索结果快照与关联素材。</p></div><div><StatusBadge status="completed" /><button onClick={() => void toggleFavorite()}>{selected.is_favorite ? "★ 已收藏" : "☆ 收藏"}</button></div></header>
                  <div className="representative-strip">{assets.slice(0, 8).map((asset, index) => <article key={asset.asset_id}><AssetThumb preview={asset.preview_url} name={asset.asset_name || asset.file_name} type={asset.asset_type} /><span>ASSET {index + 1}</span><strong>{asset.asset_name || asset.file_name}</strong></article>)}</div>
                </>
              ) : <div className="asset-empty"><strong>选择一个 Capsule 查看详情</strong></div>}
            </section>
          </div>
        )}
      </section>
    </DemoShell>
  );
}
