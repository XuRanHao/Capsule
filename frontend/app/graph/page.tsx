"use client";

import cytoscape, { Core, ElementDefinition } from "cytoscape";
import { useCallback, useEffect, useRef, useState } from "react";
import DemoShell from "../components/DemoShell";
import { apiFetch, endpoint } from "../lib/api";
import { useWorkspaceSelection, WorkspaceSelect } from "../lib/workspaces";

type MainEntity = {
  subject: string;
  description: string;
  salience: number;
  origins: string[];
};
type GraphAsset = {
  asset_id: string;
  source_path: string;
  asset_name: string;
  asset_description: string;
  main_entities: MainEntity[];
};
type GraphEntity = {
  entity_id: string;
  name: string;
  semantic: string;
  origins: string[];
  asset_ids: string[];
  descriptions: string[];
};
type GraphEdge = {
  source: string;
  target: string;
  relation: string;
  description: string;
  reason?: string;
  content_subject?: string;
};
type GraphData = {
  workspace_id: string;
  source_asset_count: number;
  understood_asset_count: number;
  understanding_errors: Array<{ asset_id?: string; error?: string }>;
  asset_count: number;
  entity_count: number;
  edge_count: number;
  assets: GraphAsset[];
  entities: GraphEntity[];
  edges: GraphEdge[];
};
type Selection =
  | { kind: "entity"; value: GraphEntity }
  | { kind: "asset"; value: GraphAsset }
  | { kind: "edge"; value: GraphEdge };

const GRAPH_READ_TIMEOUT_MS = 60_000;
const GRAPH_REBUILD_TIMEOUT_MS = 120_000;

function assetImage(asset: GraphAsset, workspaceId: string) {
  return endpoint(
    `/api/v1/assets/${encodeURIComponent(asset.asset_id)}/thumbnail?workspace_id=${encodeURIComponent(workspaceId)}`,
  );
}

function membershipEdges(graph: GraphData) {
  const entityIds = new Set(graph.entities.map((entity) => entity.entity_id));
  return graph.edges.filter(
    (edge) => entityIds.has(edge.target) && !entityIds.has(edge.source),
  );
}

function visibleGraph(graph: GraphData, hiddenNodeIds: Set<string>) {
  const candidates = membershipEdges(graph).filter(
    (edge) =>
      !hiddenNodeIds.has(edge.source) && !hiddenNodeIds.has(edge.target),
  );
  const memberCountByEntity = new Map<string, number>();
  for (const edge of candidates) {
    memberCountByEntity.set(
      edge.target,
      (memberCountByEntity.get(edge.target) ?? 0) + 1,
    );
  }
  const entityIds = new Set(
    [...memberCountByEntity]
      .filter(([, count]) => count >= 2)
      .map(([entityId]) => entityId),
  );
  const memberships = candidates.filter((edge) => entityIds.has(edge.target));
  const assetIds = new Set(memberships.map((edge) => edge.source));
  return { memberships, entityIds, assetIds };
}

function graphElements(
  graph: GraphData,
  workspaceId: string,
  hiddenNodeIds: Set<string>,
): ElementDefinition[] {
  const { memberships, entityIds, assetIds } = visibleGraph(
    graph,
    hiddenNodeIds,
  );
  const entities = graph.entities
    .filter((entity) => entityIds.has(entity.entity_id))
    .map((entity) => ({
      data: { id: entity.entity_id, label: entity.name, kind: "entity" },
    }));
  const assets = graph.assets
    .filter((asset) => assetIds.has(asset.asset_id))
    .map((asset) => ({
      data: {
        id: asset.asset_id,
        label: asset.source_path.split("/").at(-1) ?? asset.asset_name,
        kind: "asset",
        image: assetImage(asset, workspaceId),
      },
    }));
  const edges = memberships.map((edge, index) => ({
    data: {
      id: `membership-${index}`,
      source: edge.source,
      target: edge.target,
      relation: edge.relation,
      description: edge.description,
      reason: edge.reason,
    },
  }));
  return [...entities, ...assets, ...edges];
}

export default function RelationGraphPage() {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Core | null>(null);
  const buildRequestRef = useRef(0);
  const {
    workspaceId,
    workspaces,
    ready,
    loading: workspacesLoading,
    setWorkspaceId,
  } = useWorkspaceSelection();
  const [graph, setGraph] = useState<GraphData | null>(null);
  const [selection, setSelection] = useState<Selection | null>(null);
  const [hiddenNodeIds, setHiddenNodeIds] = useState<Set<string>>(new Set());
  const [building, setBuilding] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const buildGraph = useCallback(
    async (forceUnderstanding = false) => {
      if (!workspaceId) return;
      const requestId = ++buildRequestRef.current;
      let timedOut = false;
      let timeout = 0;
      setBuilding(true);
      setError(null);
      setSelection(null);
      setHiddenNodeIds(new Set());
      try {
        const query = new URLSearchParams({ workspace_id: workspaceId });
        if (forceUnderstanding) query.set("force_understanding", "true");
        const request = apiFetch<GraphData>(
          `/api/v1/relation-graphs/build?${query}`,
          { method: "POST" },
        );
        const requestTimeout = new Promise<never>((_, reject) => {
          timeout = window.setTimeout(() => {
            timedOut = true;
            reject(new Error("graph request timed out"));
          }, forceUnderstanding ? GRAPH_REBUILD_TIMEOUT_MS : GRAPH_READ_TIMEOUT_MS);
        });
        // Do not abort the streaming local proxy: leave it to finish, then
        // ignore its stale response by requestId after this UI timeout wins.
        const payload = await Promise.race([request, requestTimeout]);
        if (requestId === buildRequestRef.current) setGraph(payload);
      } catch (reason: unknown) {
        if (requestId !== buildRequestRef.current) return;
        setGraph(null);
        setError(
          timedOut
            ? "图谱请求超时，后端可能已重启。请重新读取图谱。"
            : reason instanceof Error
              ? reason.message
              : "关系图谱构建失败",
        );
      } finally {
        window.clearTimeout(timeout);
        if (requestId === buildRequestRef.current) setBuilding(false);
      }
    },
    [workspaceId],
  );

  useEffect(() => {
    if (!ready || !workspaceId) return;
    const timer = window.setTimeout(() => void buildGraph(), 0);
    return () => {
      window.clearTimeout(timer);
      buildRequestRef.current += 1;
    };
  }, [buildGraph, ready, workspaceId]);

  useEffect(() => {
    if (!graph || !containerRef.current) return;
    const assetsById = new Map(
      graph.assets.map((asset) => [asset.asset_id, asset]),
    );
    const entitiesById = new Map(
      graph.entities.map((entity) => [entity.entity_id, entity]),
    );
    const cy = cytoscape({
      container: containerRef.current,
      elements: graphElements(graph, workspaceId, hiddenNodeIds),
      layout: {
        name: "cose",
        animate: false,
        fit: true,
        padding: 55,
        nodeRepulsion: () => 900_000,
        idealEdgeLength: () => 170,
      },
      style: [
        { selector: "node[kind = 'entity']", style: { "background-color": "#6557ef", "border-color": "#fff", "border-width": 4, color: "#17151f", label: "data(label)", "font-size": 15, "font-weight": 700, "text-valign": "bottom", "text-margin-y": 12, width: 70, height: 70 } },
        { selector: "node[kind = 'asset']", style: { "background-image": "data(image)", "background-fit": "cover", "background-color": "#e8e5df", "border-color": "#fff", "border-width": 3, color: "#38333f", label: "data(label)", "font-size": 9, "text-valign": "bottom", "text-margin-y": 7, width: 54, height: 54 } },
        { selector: "edge", style: { width: 1.8, "line-color": "#a59ee5", "target-arrow-color": "#a59ee5", "target-arrow-shape": "triangle", "curve-style": "bezier", opacity: 0.78 } },
        { selector: "edge:selected", style: { width: 5, "line-color": "#f2643f", "target-arrow-color": "#f2643f", opacity: 1 } },
        { selector: ":selected", style: { "border-color": "#f2643f", "border-width": 6 } },
      ],
      minZoom: 0.35,
      maxZoom: 2.2,
    });
    cy.on("tap", "node", (event) => {
      const id = event.target.id();
      const entity = entitiesById.get(id);
      const asset = assetsById.get(id);
      if (entity) setSelection({ kind: "entity", value: entity });
      if (asset) setSelection({ kind: "asset", value: asset });
    });
    cy.on("tap", "edge", (event) => {
      const data = event.target.data();
      const edge = graph.edges.find(
        (item) =>
          item.source === data.source &&
          item.target === data.target &&
          item.relation === data.relation,
      );
      if (edge) setSelection({ kind: "edge", value: edge });
    });
    cyRef.current = cy;
    const firstVisibleEntity = graph.entities.find((entity) =>
      visibleGraph(graph, hiddenNodeIds).entityIds.has(entity.entity_id),
    );
    setSelection(
      firstVisibleEntity
        ? { kind: "entity", value: firstVisibleEntity }
        : null,
    );
    return () => {
      cy.destroy();
      cyRef.current = null;
    };
  }, [graph, hiddenNodeIds, workspaceId]);

  const visible = graph
    ? visibleGraph(graph, hiddenNodeIds)
    : { memberships: [], entityIds: new Set<string>(), assetIds: new Set<string>() };
  const memberships = visible.memberships;
  const visibleEntities = visible.entityIds.size;
  const visibleAssets = visible.assetIds.size;
  const linkedAssetIds = new Set(
    (graph ? membershipEdges(graph) : []).map((edge) => edge.source),
  );
  const unrelatedCount = graph
    ? graph.assets.filter((asset) => !linkedAssetIds.has(asset.asset_id)).length
    : 0;

  const hideSelectedNode = () => {
    if (!selection || selection.kind === "edge") return;
    const nodeId =
      selection.kind === "entity"
        ? selection.value.entity_id
        : selection.value.asset_id;
    setHiddenNodeIds((current) => new Set(current).add(nodeId));
    setSelection(null);
  };

  return (
    <DemoShell
      active="graph"
      workspaceControl={
        <WorkspaceSelect
          workspaceId={workspaceId}
          workspaces={workspaces}
          loading={workspacesLoading}
          onChange={(nextWorkspaceId) => {
            buildRequestRef.current += 1;
            setGraph(null);
            setSelection(null);
            setHiddenNodeIds(new Set());
            setError(null);
            setBuilding(false);
            setWorkspaceId(nextWorkspaceId);
          }}
        />
      }
      eyebrow="RELATION GRAPH / LIVE WORKSPACE"
      title="元数据与内容，分路提取后合并。"
      description="从当前工作空间读取 Asset，补齐内容理解，再由 Agent 判断节点关系并生成描述。"
      actions={
        <button
          className="primary-action"
          disabled={building || !ready}
          onClick={() => void buildGraph(true)}
        >
          {building ? "正在构建…" : "重新理解并建图"}
        </button>
      }
    >
      <section className="graph-metrics" aria-label="图谱统计">
        <article><small>ASSETS</small><strong>{visibleAssets}</strong><span>当前可见</span></article>
        <article><small>SHARED ENTITIES</small><strong>{visibleEntities}</strong><span>至少连接 2 个 Asset</span></article>
        <article><small>MEMBERSHIPS</small><strong>{memberships.length}</strong><span>Asset → Entity</span></article>
      </section>
      <section className="graph-finding-strip">
        <span>节点准入</span><strong>{visibleEntities} 个共享 Entity</strong><i /><strong>{unrelatedCount} 个无连接 Asset 已隐藏</strong><i /><strong>{hiddenNodeIds.size} 个节点手动隐藏</strong>
        <p>画布只显示参与关系的节点；点击连线可查看 Agent 生成的关系类型和描述。</p>
      </section>
      <section className="relation-graph-shell">
        <header className="relation-toolbar">
          <div><strong>Cytoscape 自动布局</strong><span>{graph ? `工作空间 ${graph.workspace_id}` : "等待工作空间数据"}</span></div>
          <div className="relation-filters">
            {hiddenNodeIds.size > 0 && (
              <button onClick={() => setHiddenNodeIds(new Set())}>恢复全部</button>
            )}
            <button onClick={() => cyRef.current?.fit(undefined, 55)}>适应画布</button>
          </div>
        </header>
        <div className="relation-workspace">
          <div className="relation-canvas-wrap">
            {error && <div className="asset-empty"><strong>图谱构建失败</strong><span>{error}</span><button className="secondary-action" onClick={() => void buildGraph()}>重新读取图谱</button></div>}
            {building && <div className="asset-empty"><strong>正在读取 Asset 并构建关系图谱…</strong></div>}
            {!building && graph && visibleEntities === 0 && <div className="asset-empty"><strong>没有可见的共享实体</strong><span>恢复隐藏节点，或切换包含更多关系的工作空间。</span></div>}
            <div ref={containerRef} className="cytoscape-canvas" />
            <footer className="relation-legend"><span><i className="solid" />Asset → 共享 Entity</span></footer>
          </div>
          <aside className="relation-inspector">
            <span className="eyebrow">NODE INSPECTOR</span>
            {!selection && <p>点击节点或连线查看详情。</p>}
            {selection?.kind === "entity" && <><small>SHARED ENTITY</small><h2>{selection.value.name}</h2><button className="secondary-action relation-hide-node" onClick={hideSelectedNode}>隐藏节点</button><p>{selection.value.semantic}</p><p>连接 {selection.value.asset_ids.length} 个 Asset。</p><div className="relation-inspector-list">{selection.value.asset_ids.filter((assetId) => visible.assetIds.has(assetId)).map((assetId) => <button key={assetId} onClick={() => cyRef.current?.getElementById(assetId).select()}><span>{graph?.assets.find((asset) => asset.asset_id === assetId)?.source_path}</span></button>)}</div></>}
            {selection?.kind === "asset" && <><small>ASSET</small><h2>{selection.value.asset_name}</h2>
              <button className="secondary-action relation-hide-node" onClick={hideSelectedNode}>隐藏节点</button>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img className="inspector-preview" src={assetImage(selection.value, workspaceId)} alt={selection.value.source_path} /><p>{selection.value.asset_description}</p><div className="secondary-subjects">{selection.value.main_entities.map((entity) => <span key={entity.subject}>{entity.subject}</span>)}</div></>}
            {selection?.kind === "edge" && <><small>AGENT RELATION</small><h2>{selection.value.relation}</h2><dl><div><dt>素材</dt><dd>{graph?.assets.find((asset) => asset.asset_id === selection.value.source)?.source_path ?? selection.value.source}</dd></div><div><dt>内容主体</dt><dd>{selection.value.content_subject || "—"}</dd></div><div><dt>实体</dt><dd>{graph?.entities.find((entity) => entity.entity_id === selection.value.target)?.name ?? selection.value.target}</dd></div></dl><p>{selection.value.description || "Agent 未返回关系描述。"}</p>{selection.value.reason && <><small>判断依据</small><p>{selection.value.reason}</p></>}</>}
          </aside>
        </div>
      </section>
    </DemoShell>
  );
}
