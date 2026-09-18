"use client";

import { useCallback, useEffect, useState } from "react";
import { ProductTopbar } from "./components/DemoShell";
import AgentChat from "./components/workbench/AgentChat";
import GraphCanvas from "./components/workbench/GraphCanvas";
import WorkspacePanel from "./components/workbench/WorkspacePanel";
import {
  createNarrativeGraph,
  loadNarrativeGraphs,
  type AssetRecord,
  type NarrativeGraphRecord,
  loadAssets,
} from "./lib/api";
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
  const [activeGraph, setActiveGraph] = useState<NarrativeGraphRecord | null>(null);
  const [graphs, setGraphs] = useState<NarrativeGraphRecord[]>([]);
  const [creatingGraph, setCreatingGraph] = useState(false);
  const [graphError, setGraphError] = useState<{ workspaceId: string; message: string } | null>(null);
  const selectedGraph = activeGraph?.workspace_id === workspaceId ? activeGraph : null;
  const selectedGraphError = graphError?.workspaceId === workspaceId ? graphError.message : null;

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

  const loadWorkspaceGraphs = useCallback(async () => {
    if (!ready || !workspaceId) return;
    try {
      const items = await loadNarrativeGraphs(workspaceId);
      setGraphs(items);
      setGraphError(null);
      setActiveGraph((current) =>
        current && items.some((graph) => graph.graph_id === current.graph_id)
          ? current
          : null,
      );
    } catch (error) {
      setGraphs([]);
      setGraphError({
        workspaceId,
        message: error instanceof Error ? error.message : "图谱加载失败",
      });
    }
  }, [ready, workspaceId]);

  useEffect(() => {
    const timer = window.setTimeout(() => { void loadWorkspaceGraphs(); }, 0);
    return () => window.clearTimeout(timer);
  }, [loadWorkspaceGraphs]);

  const createGraph = useCallback(async () => {
    if (!workspaceId || creatingGraph) return;
    setCreatingGraph(true);
    setGraphError(null);
    try {
      const graph = await createNarrativeGraph({ workspace_id: workspaceId });
      setGraphs((current) => [...current, graph]);
      setActiveGraph(graph);
    } catch (error) {
      setGraphError({
        workspaceId,
        message: error instanceof Error ? error.message : "创建图谱失败",
      });
    } finally {
      setCreatingGraph(false);
    }
  }, [creatingGraph, workspaceId]);

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
          graphs={graphs}
          selectedGraphId={selectedGraph?.graph_id ?? null}
          onWorkspaceChange={setWorkspaceId}
          creatingGraph={creatingGraph}
          onCreateGraph={createGraph}
          onGraphSelect={setActiveGraph}
          onAssetsRefresh={() => { void loadWorkspaceAssets(); }}
        />
        <section className="workbench-main">
          {(workspaceError || assetsError) && (
            <div className="workbench-notice" role="status">
              <strong>暂未连接到数据服务</strong>
              <span>{workspaceError || assetsError}。页面仍可用于检查布局；启动后端后会自动加载真实素材。</span>
            </div>
          )}
          <GraphCanvas
            assets={assets}
            selectedId={selectedAssetId}
            onSelect={setSelectedAssetId}
            activeGraph={selectedGraph}
            graphError={selectedGraphError}
          />
        </section>
        <AgentChat
          key={`${workspaceId}:${selectedGraph?.graph_id ?? "no-graph"}`}
          workspaceId={workspaceId}
          selectedGraphId={selectedGraph?.graph_id ?? null}
        />
      </div>
    </main>
  );
}
