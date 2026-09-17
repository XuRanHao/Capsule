"use client";

import { useCallback, useEffect, useState } from "react";
import { ProductTopbar } from "./components/DemoShell";
import AgentChat from "./components/workbench/AgentChat";
import GraphCanvas from "./components/workbench/GraphCanvas";
import WorkspacePanel from "./components/workbench/WorkspacePanel";
import { type AssetRecord, loadAssets } from "./lib/api";
import { useWorkspaceSelection, WorkspaceSelect } from "./lib/workspaces";

export default function WorkspacePage() {
  const {
    workspaceId,
    workspaces,
    ready,
    loading: workspacesLoading,
    error: workspaceError,
    setWorkspaceId,
  } = useWorkspaceSelection();
  const [assets, setAssets] = useState<AssetRecord[]>([]);
  const [assetsLoading, setAssetsLoading] = useState(false);
  const [assetsError, setAssetsError] = useState<string | null>(null);
  const [selectedAssetId, setSelectedAssetId] = useState<string | null>(null);

  const loadWorkspaceAssets = useCallback(async () => {
    if (!ready || !workspaceId) return;
    setAssetsLoading(true);
    try {
      const params = new URLSearchParams({ workspace_id: workspaceId, limit: "100", offset: "0" });
      const result = await loadAssets(params);
      setAssets(result.items);
      setSelectedAssetId((current) => current && result.items.some((asset) => asset.asset_id === current) ? current : null);
      setAssetsError(null);
    } catch (error) {
      setAssets([]);
      setAssetsError(error instanceof Error ? error.message : "素材加载失败");
    } finally {
      setAssetsLoading(false);
    }
  }, [ready, workspaceId]);

  useEffect(() => {
    const timer = window.setTimeout(() => { void loadWorkspaceAssets(); }, 0);
    return () => window.clearTimeout(timer);
  }, [loadWorkspaceAssets]);

  return (
    <main className="workbench-shell">
      <ProductTopbar
        active="workspace"
        connection={workspaceError || assetsError ? "demo" : "live"}
        status={assetsLoading ? "正在同步素材" : assetsError ? "等待后端连接" : "工作空间已就绪"}
        workspaceControl={<WorkspaceSelect workspaceId={workspaceId} workspaces={workspaces} loading={workspacesLoading} onChange={setWorkspaceId} />}
      />
      <div className="workbench-layout">
        <WorkspacePanel
          workspaceId={workspaceId}
          workspaces={workspaces}
          loading={workspacesLoading}
          assets={assets}
          onWorkspaceChange={setWorkspaceId}
        />
        <section className="workbench-main">
          {(workspaceError || assetsError) && (
            <div className="workbench-notice" role="status">
              <strong>暂未连接到数据服务</strong>
              <span>{workspaceError || assetsError}。页面仍可用于检查布局；启动后端后会自动加载真实素材。</span>
            </div>
          )}
          <GraphCanvas assets={assets} selectedId={selectedAssetId} onSelect={setSelectedAssetId} />
        </section>
        <AgentChat key={workspaceId} workspaceId={workspaceId} selectedAssetId={selectedAssetId} />
      </div>
    </main>
  );
}
