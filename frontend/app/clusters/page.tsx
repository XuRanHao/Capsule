"use client";

import Link from "next/link";
import {
  type CSSProperties,
  type PointerEvent as ReactPointerEvent,
  type WheelEvent as ReactWheelEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import DemoShell, {
  AssetThumb,
  StatusBadge,
} from "../components/DemoShell";
import {
  ApiRequestError,
  type ClusterAssetStatus,
  type ClusterAssetStatusItem,
  type ClusterCapsule,
  type ClusterMember,
  type ClusterRun,
  type CurrentCluster,
  type CurrentClusterMember,
  type CurrentClusterMode,
  apiFetch,
} from "../lib/api";
import { useWorkspaceSelection, WorkspaceSelect } from "../lib/workspaces";

const FEATURE_TYPES = [
  { value: "native_multimodal", label: "原始内容" },
  { value: "asset_description", label: "素材完整描述" },
  { value: "subject_content", label: "主体与内容" },
  { value: "scene_theme", label: "场景与题材" },
  { value: "visual_style", label: "视觉风格" },
  { value: "color_composition", label: "色彩与构图" },
  { value: "mood_atmosphere", label: "画面情绪氛围" },
  {
    value: "character_state_or_psychology",
    label: "人物状态或心理",
  },
  { value: "asset_usage", label: "资产用途" },
  { value: "target_audience", label: "目标受众" },
  { value: "provenance", label: "来源与创作关系" },
  {
    value: "rights_version_authorship",
    label: "权利、版本与作者",
  },
] as const;

const GROUP_COLORS = [
  "#6557ef",
  "#f2643f",
  "#359f7a",
  "#bd9618",
  "#dd65aa",
  "#2b93b8",
  "#8b6d52",
  "#70a53e",
];

const GRAPH_MIN_SCALE = 0.35;
const GRAPH_MAX_SCALE = 2.4;
const GRAPH_MEMBER_LIMIT = 32;
const RUN_POLL_INTERVAL_MS = 1_000;
const ACTIVE_RUN_STATUSES = new Set(["pending", "running"]);

const CURRENT_CLUSTER_MODES: Array<{
  value: CurrentClusterMode;
  label: string;
  help: string;
}> = [
  {
    value: "dynamic",
    label: "动态簇",
    help: "参与下一次全量聚类",
  },
  {
    value: "resident_open",
    label: "开放常驻",
    help: "保留成员，并允许算法加入",
  },
  {
    value: "resident_manual",
    label: "手动管理",
    help: "簇内成员由用户手动管理",
  },
];

function currentClusterModeLabel(mode: CurrentClusterMode) {
  return (
    CURRENT_CLUSTER_MODES.find((item) => item.value === mode)?.label ?? mode
  );
}

function clusterAssetStatusLabel(status: ClusterAssetStatusItem["status"]) {
  return {
    incrementally_clustered: "已增量归簇",
    pending: "待聚类",
    manual_management: "手动管理",
  }[status];
}

function parseAssetIds(value: string) {
  return [...new Set(value.split(/[\s,，]+/).map((item) => item.trim()))].filter(
    Boolean,
  );
}

function CurrentMemberThumbnail({
  assetId,
  workspaceId,
}: {
  assetId: string;
  workspaceId: string;
}) {
  const [failed, setFailed] = useState(false);
  const previewUrl = `/api/v1/assets/${encodeURIComponent(assetId)}/thumbnail?workspace_id=${encodeURIComponent(workspaceId)}`;

  return (
    <span className="current-member-thumbnail">
      {failed ? (
        <small>NO PREVIEW</small>
      ) : (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={previewUrl}
          alt={`${assetId} 缩略图`}
          loading="lazy"
          onError={() => setFailed(true)}
        />
      )}
    </span>
  );
}

type GraphMemberNode = {
  member: ClusterMember;
  x: number;
  y: number;
};

type GraphGroup = {
  capsule: ClusterCapsule;
  color: string;
  diameter: number;
  left: number;
  top: number;
  members: ClusterMember[];
  memberNodes: GraphMemberNode[];
};

type GraphLayout = {
  width: number;
  height: number;
  groups: GraphGroup[];
};

type GraphViewport = {
  scale: number;
  x: number;
  y: number;
};

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(maximum, Math.max(minimum, value));
}

function parseIntegerParameter(
  label: string,
  value: string,
  minimum: number,
  maximum: number,
) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < minimum || parsed > maximum) {
    throw new Error(`${label} 必须是 ${minimum} 到 ${maximum} 之间的整数`);
  }
  return parsed;
}

function runParameter(value: unknown) {
  return typeof value === "number" ? value : "—";
}

function runPcaDimension(run: ClusterRun | undefined) {
  return runParameter(
    run?.preprocessing.pca_dimension ??
      run?.preprocessing.requested_pca_dimension,
  );
}

function runOptionLabel(run: ClusterRun) {
  const feature =
    FEATURE_TYPES.find((item) => item.value === run.embedding_type)?.label ??
    run.embedding_type;
  const pca = runPcaDimension(run);
  const minSamples = runParameter(run.parameters.min_samples);
  const minClusterSize = runParameter(run.parameters.min_cluster_size);
  return `${feature} · ${run.status} · ${run.sample_count} 条 · PCA ${pca} / MS ${minSamples} / MCS ${minClusterSize}`;
}

function isActiveRun(run: ClusterRun | undefined) {
  return Boolean(run && ACTIVE_RUN_STATUSES.has(run.status));
}

function placeMemberNodes(
  members: ClusterMember[],
  radius: number,
): GraphMemberNode[] {
  const visibleMembers = members.slice(0, GRAPH_MEMBER_LIMIT);
  return visibleMembers.map((member, index) => {
    const progress = Math.sqrt((index + 1) / (visibleMembers.length + 1));
    const distance = radius * (0.48 + progress * 0.32);
    const angle = index * 2.3999632297 - Math.PI / 2;
    return {
      member,
      x: radius + Math.cos(angle) * distance,
      y: radius + Math.sin(angle) * distance,
    };
  });
}

function ClusterKnowledgeGraph({
  layout,
  selectedCapsuleId,
  emptyMessage,
  onSelect,
}: {
  layout: GraphLayout;
  selectedCapsuleId: string;
  emptyMessage: string;
  onSelect: (capsuleId: string) => void;
}) {
  const viewportElement = useRef<HTMLDivElement>(null);
  const dragState = useRef<{
    pointerId: number;
    clientX: number;
    clientY: number;
  } | null>(null);
  const [viewport, setViewport] = useState<GraphViewport>({
    scale: 1,
    x: 0,
    y: 0,
  });
  const [dragging, setDragging] = useState(false);

  const fitGraph = useCallback(() => {
    const element = viewportElement.current;
    if (!element) return;
    const bounds = element.getBoundingClientRect();
    const scale = clamp(
      Math.min(
        (bounds.width - 56) / layout.width,
        (bounds.height - 56) / layout.height,
      ),
      GRAPH_MIN_SCALE,
      1.08,
    );
    setViewport({
      scale,
      x: (bounds.width - layout.width * scale) / 2,
      y: (bounds.height - layout.height * scale) / 2,
    });
  }, [layout.height, layout.width]);

  useEffect(() => {
    const animationFrame = window.requestAnimationFrame(fitGraph);
    window.addEventListener("resize", fitGraph);
    return () => {
      window.cancelAnimationFrame(animationFrame);
      window.removeEventListener("resize", fitGraph);
    };
  }, [fitGraph]);

  const zoomAt = useCallback(
    (nextScale: number, originX?: number, originY?: number) => {
      const element = viewportElement.current;
      if (!element) return;
      const bounds = element.getBoundingClientRect();
      const x = originX ?? bounds.width / 2;
      const y = originY ?? bounds.height / 2;
      setViewport((current) => {
        const scale = clamp(nextScale, GRAPH_MIN_SCALE, GRAPH_MAX_SCALE);
        const ratio = scale / current.scale;
        return {
          scale,
          x: x - (x - current.x) * ratio,
          y: y - (y - current.y) * ratio,
        };
      });
    },
    [],
  );

  const handleWheel = (event: ReactWheelEvent<HTMLDivElement>) => {
    event.preventDefault();
    const bounds = event.currentTarget.getBoundingClientRect();
    const scale = viewport.scale * Math.exp(-event.deltaY * 0.0012);
    zoomAt(scale, event.clientX - bounds.left, event.clientY - bounds.top);
  };

  const handlePointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (
      event.button !== 0 ||
      (event.target as HTMLElement).closest(".cluster-graph-node")
    ) {
      return;
    }
    event.currentTarget.setPointerCapture(event.pointerId);
    dragState.current = {
      pointerId: event.pointerId,
      clientX: event.clientX,
      clientY: event.clientY,
    };
    setDragging(true);
  };

  const handlePointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    const drag = dragState.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const deltaX = event.clientX - drag.clientX;
    const deltaY = event.clientY - drag.clientY;
    drag.clientX = event.clientX;
    drag.clientY = event.clientY;
    setViewport((current) => ({
      ...current,
      x: current.x + deltaX,
      y: current.y + deltaY,
    }));
  };

  const endDrag = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (dragState.current?.pointerId !== event.pointerId) return;
    dragState.current = null;
    setDragging(false);
  };

  return (
    <div className="cluster-graph-shell">
      <div className="cluster-graph-toolbar">
        <span>滚轮缩放 · 拖拽画布 · 双击适应</span>
        <div role="group" aria-label="知识图谱缩放控制">
          <button
            type="button"
            aria-label="缩小知识图谱"
            disabled={viewport.scale <= GRAPH_MIN_SCALE}
            onClick={() => zoomAt(viewport.scale / 1.2)}
          >
            −
          </button>
          <output aria-label="当前缩放比例">
            {Math.round(viewport.scale * 100)}%
          </output>
          <button
            type="button"
            aria-label="放大知识图谱"
            disabled={viewport.scale >= GRAPH_MAX_SCALE}
            onClick={() => zoomAt(viewport.scale * 1.2)}
          >
            ＋
          </button>
          <button type="button" className="fit" onClick={fitGraph}>
            适应
          </button>
        </div>
      </div>
      <div
        ref={viewportElement}
        className={`cluster-graph-viewport ${dragging ? "dragging" : ""}`}
        aria-label="可缩放、可拖拽的聚类知识图谱"
        onDoubleClick={fitGraph}
        onPointerCancel={endDrag}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={endDrag}
        onWheel={handleWheel}
      >
        <div
          className="cluster-graph-world"
          style={{
            width: layout.width,
            height: layout.height,
            transform: `translate3d(${viewport.x}px, ${viewport.y}px, 0) scale(${viewport.scale})`,
          }}
        >
          {layout.groups.map((group, index) => {
            const selected =
              selectedCapsuleId === group.capsule.cluster_capsule_id;
            const hiddenMemberCount = Math.max(
              0,
              group.capsule.member_count - group.memberNodes.length,
            );
            return (
              <button
                type="button"
                className={`cluster-graph-node ${selected ? "active" : ""}`}
                aria-label={`${group.capsule.effective_name}，${group.capsule.member_count} 个资产`}
                aria-pressed={selected}
                style={
                  {
                    "--cluster-color": group.color,
                    left: group.left,
                    top: group.top,
                    width: group.diameter,
                    height: group.diameter,
                  } as CSSProperties
                }
                onClick={() =>
                  onSelect(group.capsule.cluster_capsule_id)
                }
                key={group.capsule.cluster_capsule_id}
              >
                <svg
                  className="cluster-node-network"
                  viewBox={`0 0 ${group.diameter} ${group.diameter}`}
                  aria-hidden="true"
                >
                  {group.memberNodes.map(({ member, x, y }) => (
                    <g key={member.asset_id}>
                      <line
                        x1={group.diameter / 2}
                        y1={group.diameter / 2}
                        x2={x}
                        y2={y}
                      />
                      <circle
                        className={
                          member.asset_id === group.capsule.medoid_asset_id
                            ? "medoid"
                            : ""
                        }
                        cx={x}
                        cy={y}
                        r={
                          member.asset_id === group.capsule.medoid_asset_id
                            ? 7
                            : 5
                        }
                      >
                        <title>{member.asset_name || member.file_name}</title>
                      </circle>
                    </g>
                  ))}
                </svg>
                <span className="cluster-node-index">
                  {String(index + 1).padStart(2, "0")}
                </span>
                <span className="cluster-node-label">
                  <strong>{group.capsule.effective_name}</strong>
                  <small>{group.capsule.member_count} ASSETS</small>
                </span>
                {hiddenMemberCount > 0 && (
                  <span className="cluster-node-overflow">
                    +{hiddenMemberCount}
                  </span>
                )}
              </button>
            );
          })}
        </div>
        {!layout.groups.length && (
          <div className="cluster-map-empty">{emptyMessage}</div>
        )}
      </div>
    </div>
  );
}

export default function ClustersPage() {
  const {
    workspaceId,
    workspaces,
    ready: workspaceReady,
    loading: workspacesLoading,
    setWorkspaceId,
  } = useWorkspaceSelection();
  const [runs, setRuns] = useState<ClusterRun[]>([]);
  const [selectedRunId, setSelectedRunId] = useState("");
  const [embeddingType, setEmbeddingType] = useState<string>(
    FEATURE_TYPES[0].value,
  );
  const [pcaDimension, setPcaDimension] = useState("8");
  const [minSamples, setMinSamples] = useState("3");
  const [minClusterSize, setMinClusterSize] = useState("3");
  const [capsules, setCapsules] = useState<ClusterCapsule[]>([]);
  const [selectedCapsuleId, setSelectedCapsuleId] = useState("");
  const [membersByCapsule, setMembersByCapsule] = useState<
    Record<string, ClusterMember[]>
  >({});
  const [currentClusters, setCurrentClusters] = useState<CurrentCluster[]>([]);
  const [assetStatus, setAssetStatus] = useState<ClusterAssetStatus | null>(
    null,
  );
  const [assetStatusLoading, setAssetStatusLoading] = useState(true);
  const [assetStatusError, setAssetStatusError] = useState<string | null>(null);
  const [selectedCurrentClusterId, setSelectedCurrentClusterId] = useState("");
  const [currentMembers, setCurrentMembers] = useState<CurrentClusterMember[]>(
    [],
  );
  const [currentLoading, setCurrentLoading] = useState(true);
  const [currentMutation, setCurrentMutation] = useState("");
  const [currentError, setCurrentError] = useState<string | null>(null);
  const [currentNotice, setCurrentNotice] = useState<string | null>(null);
  const [assetIdsInput, setAssetIdsInput] = useState("");
  const [moveTargets, setMoveTargets] = useState<Record<string, string>>({});
  const [running, setRunning] = useState(false);
  const [runNotice, setRunNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const runRequestRef = useRef(0);
  const currentClusterRequestRef = useRef(0);
  const assetStatusRequestRef = useRef(0);
  const currentMemberRequestRef = useRef(0);
  const currentClusterWorkspaceRef = useRef<HTMLElement>(null);
  const capsuleCardRefs = useRef(new Map<string, HTMLElement>());
  const clusterDetailRef = useRef<HTMLElement>(null);
  const deepLinkTargetRef = useRef(
    typeof window === "undefined"
      ? { clusterRunId: "", clusterCapsuleId: "" }
      : (() => {
          const searchParams = new URLSearchParams(window.location.search);
          return {
            clusterRunId:
              searchParams.get("cluster_run_id")?.trim() || "",
            clusterCapsuleId:
              searchParams.get("cluster_capsule_id")?.trim() || "",
          };
        })(),
  );
  const deepLinkOpenedRef = useRef(false);

  const loadRuns = useCallback(async () => {
    if (!workspaceReady) return;
    const requestId = ++runRequestRef.current;
    try {
      const payload = await apiFetch<{ items: ClusterRun[] }>(
        `/api/v1/cluster-runs?workspace_id=${encodeURIComponent(workspaceId)}&limit=100`,
      );
      if (requestId !== runRequestRef.current) return;
      setRuns(payload.items);
      setSelectedRunId((current) => {
        const deepLinkedRunId = deepLinkTargetRef.current.clusterRunId;
        if (
          !deepLinkOpenedRef.current &&
          deepLinkedRunId &&
          payload.items.some((run) => run.cluster_run_id === deepLinkedRunId)
        ) {
          return deepLinkedRunId;
        }
        return current &&
          payload.items.some((run) => run.cluster_run_id === current)
          ? current
          : payload.items[0]?.cluster_run_id || "";
      });
      setError(null);
    } catch (requestError) {
      if (requestId !== runRequestRef.current) return;
      setError(
        requestError instanceof Error ? requestError.message : "聚类记录加载失败",
      );
    }
  }, [workspaceId, workspaceReady]);

  useEffect(() => {
    const initial = window.setTimeout(() => void loadRuns(), 0);
    return () => window.clearTimeout(initial);
  }, [loadRuns]);

  const loadCurrentClusters = useCallback(async () => {
    if (!workspaceReady) return;
    const requestId = ++currentClusterRequestRef.current;
    setCurrentLoading(true);
    try {
      const params = new URLSearchParams({
        workspace_id: workspaceId,
        embedding_type: embeddingType,
      });
      const payload = await apiFetch<{ items: CurrentCluster[] }>(
        `/api/v1/clusters?${params}`,
      );
      if (requestId !== currentClusterRequestRef.current) return;
      setCurrentClusters(payload.items);
      setSelectedCurrentClusterId((current) =>
        payload.items.some((cluster) => cluster.cluster_id === current)
          ? current
          : payload.items[0]?.cluster_id || "",
      );
      setCurrentError(null);
    } catch (requestError) {
      if (requestId !== currentClusterRequestRef.current) return;
      setCurrentClusters([]);
      setSelectedCurrentClusterId("");
      setCurrentError(
        requestError instanceof Error
          ? requestError.message
          : "当前聚类加载失败",
      );
    } finally {
      if (requestId === currentClusterRequestRef.current) {
        setCurrentLoading(false);
      }
    }
  }, [embeddingType, workspaceId, workspaceReady]);

  const loadAssetStatus = useCallback(async () => {
    if (!workspaceReady) return;
    const requestId = ++assetStatusRequestRef.current;
    setAssetStatusLoading(true);
    try {
      const params = new URLSearchParams({
        workspace_id: workspaceId,
        embedding_type: embeddingType,
      });
      const payload = await apiFetch<ClusterAssetStatus>(
        `/api/v1/clusters/assets/status?${params}`,
      );
      if (requestId !== assetStatusRequestRef.current) return;
      setAssetStatus(payload);
      setAssetStatusError(null);
    } catch (requestError) {
      if (requestId !== assetStatusRequestRef.current) return;
      setAssetStatus(null);
      setAssetStatusError(
        requestError instanceof Error
          ? requestError.message
          : "Asset 聚类状态加载失败",
      );
    } finally {
      if (requestId === assetStatusRequestRef.current) {
        setAssetStatusLoading(false);
      }
    }
  }, [embeddingType, workspaceId, workspaceReady]);

  const refreshClusterDimension = useCallback(async () => {
    await Promise.all([loadCurrentClusters(), loadAssetStatus()]);
  }, [loadAssetStatus, loadCurrentClusters]);

  useEffect(() => {
    const initial = window.setTimeout(() => {
      setCurrentClusters([]);
      setAssetStatus(null);
      setAssetStatusError(null);
      setSelectedCurrentClusterId("");
      setCurrentMembers([]);
      setCurrentNotice(null);
      setAssetIdsInput("");
      void refreshClusterDimension();
    }, 0);
    return () => window.clearTimeout(initial);
  }, [refreshClusterDimension]);

  const selectedCurrentCluster =
    currentClusters.find(
      (cluster) => cluster.cluster_id === selectedCurrentClusterId,
    ) ?? currentClusters[0];
  const activeCurrentClusterId = selectedCurrentCluster?.cluster_id ?? "";
  const availableMoveTargets = currentClusters.filter(
    (cluster) =>
      cluster.cluster_id !== selectedCurrentCluster?.cluster_id &&
      cluster.mode !== "dynamic",
  );

  const loadCurrentMembers = useCallback(async () => {
    const requestId = ++currentMemberRequestRef.current;
    setCurrentMembers([]);
    if (!activeCurrentClusterId) {
      return;
    }
    try {
      const params = new URLSearchParams({ workspace_id: workspaceId });
      const payload = await apiFetch<{ items: CurrentClusterMember[] }>(
        `/api/v1/clusters/${encodeURIComponent(activeCurrentClusterId)}/members?${params}`,
      );
      if (requestId === currentMemberRequestRef.current) {
        setCurrentMembers(payload.items);
        setCurrentError(null);
      }
    } catch (requestError) {
      if (requestId !== currentMemberRequestRef.current) return;
      setCurrentMembers([]);
      if (
        requestError instanceof ApiRequestError &&
        requestError.status === 404 &&
        requestError.code === "current_cluster_not_found"
      ) {
        setCurrentError(null);
        await loadCurrentClusters();
        return;
      }
      setCurrentError(
        requestError instanceof Error ? requestError.message : "簇成员加载失败",
      );
    }
  }, [activeCurrentClusterId, loadCurrentClusters, workspaceId]);

  useEffect(() => {
    const timer = window.setTimeout(() => void loadCurrentMembers(), 0);
    return () => window.clearTimeout(timer);
  }, [loadCurrentMembers]);

  const selectedRun =
    runs.find((run) => run.cluster_run_id === selectedRunId) ?? runs[0];

  const activeRunForDimension = runs.find(
    (run) => run.embedding_type === embeddingType && isActiveRun(run),
  );

  useEffect(() => {
    if (!isActiveRun(selectedRun)) return;

    let cancelled = false;
    let pollTimer: number | undefined;

    const pollRun = async () => {
      try {
        const run = await apiFetch<ClusterRun>(
          `/api/v1/cluster-runs/${encodeURIComponent(selectedRun.cluster_run_id)}?workspace_id=${encodeURIComponent(workspaceId)}`,
        );
        if (cancelled) return;

        setRuns((current) => {
          const exists = current.some(
            (item) => item.cluster_run_id === run.cluster_run_id,
          );
          return exists
            ? current.map((item) =>
                item.cluster_run_id === run.cluster_run_id ? run : item,
              )
            : [run, ...current];
        });

        if (isActiveRun(run)) {
          setRunNotice(
            run.status === "pending"
              ? "全量重聚类已提交，等待开始计算…"
              : "全量重聚类正在计算，完成后会自动展示结果…",
          );
          pollTimer = window.setTimeout(pollRun, RUN_POLL_INTERVAL_MS);
          return;
        }

        if (run.status === "completed") {
          setRunNotice(
            `全量重聚类完成：${run.cluster_count ?? 0} 个簇，${run.noise_count ?? 0} 个噪声样本。`,
          );
          setError(null);
        } else if (run.status === "insufficient_data") {
          setRunNotice("全量重聚类完成，但当前维度样本不足，未生成簇。");
        } else if (run.status === "failed") {
          const reason =
            typeof run.preprocessing.error === "string"
              ? `：${run.preprocessing.error}`
              : "";
          setRunNotice(null);
          setError(`全量重聚类失败${reason}`);
        }
        await refreshClusterDimension();
      } catch (requestError) {
        if (cancelled) return;
        setError(
          requestError instanceof Error
            ? `聚类状态刷新失败：${requestError.message}`
            : "聚类状态刷新失败",
        );
        pollTimer = window.setTimeout(pollRun, RUN_POLL_INTERVAL_MS);
      }
    };

    pollTimer = window.setTimeout(pollRun, RUN_POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      if (pollTimer !== undefined) window.clearTimeout(pollTimer);
    };
  }, [refreshClusterDimension, selectedRun, workspaceId]);

  useEffect(() => {
    let cancelled = false;
    const timer = window.setTimeout(() => {
      if (
        !selectedRun ||
        !["completed", "insufficient_data"].includes(selectedRun.status)
      ) {
        setCapsules([]);
        setMembersByCapsule({});
        return;
      }
      void apiFetch<{ items: ClusterCapsule[] }>(
        `/api/v1/cluster-runs/${selectedRun.cluster_run_id}/capsules?workspace_id=${encodeURIComponent(workspaceId)}`,
      )
        .then(async (payload) => {
          if (cancelled) return;
          setCapsules(payload.items);
          setSelectedCapsuleId((current) => {
            const deepLinkedCapsuleId =
              deepLinkTargetRef.current.clusterCapsuleId;
            if (
              deepLinkedCapsuleId &&
              payload.items.some(
                (capsule) =>
                  capsule.cluster_capsule_id === deepLinkedCapsuleId,
              )
            ) {
              return deepLinkedCapsuleId;
            }
            return current &&
              payload.items.some(
                (capsule) => capsule.cluster_capsule_id === current,
              )
              ? current
              : payload.items[0]?.cluster_capsule_id || "";
          });
          const entries = await Promise.all(
            payload.items.map(async (capsule) => {
              const members = await apiFetch<{ items: ClusterMember[] }>(
                `/api/v1/cluster-capsules/${capsule.cluster_capsule_id}/members?workspace_id=${encodeURIComponent(workspaceId)}`,
              );
              return [capsule.cluster_capsule_id, members.items] as const;
            }),
          );
          if (cancelled) return;
          setMembersByCapsule(Object.fromEntries(entries));
        })
        .catch((requestError: unknown) => {
          if (cancelled) return;
          setError(
            requestError instanceof Error
              ? requestError.message
              : "Cluster Capsule 加载失败",
          );
        });
    }, 0);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [selectedRun, workspaceId]);

  const selectedCapsule =
    capsules.find(
      (capsule) => capsule.cluster_capsule_id === selectedCapsuleId,
    ) ?? capsules[0];
  const selectedMembers = selectedCapsule
    ? membersByCapsule[selectedCapsule.cluster_capsule_id] || []
    : [];

  useEffect(() => {
    if (!selectedCapsuleId) return;
    capsuleCardRefs.current
      .get(selectedCapsuleId)
      ?.scrollIntoView({ block: "nearest" });
  }, [selectedCapsuleId]);

  useEffect(() => {
    const deepLinkedCapsuleId = deepLinkTargetRef.current.clusterCapsuleId;
    if (
      !deepLinkedCapsuleId ||
      deepLinkOpenedRef.current ||
      selectedCapsule?.cluster_capsule_id !== deepLinkedCapsuleId
    ) {
      return;
    }
    deepLinkOpenedRef.current = true;
    window.requestAnimationFrame(() => {
      clusterDetailRef.current?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    });
  }, [selectedCapsule?.cluster_capsule_id]);

  const createRun = async () => {
    if (activeRunForDimension) {
      setSelectedRunId(activeRunForDimension.cluster_run_id);
      setRunNotice("当前维度已有全量重聚类任务，正在等待它完成…");
      return;
    }
    setRunning(true);
    setError(null);
    setRunNotice("正在提交全量重聚类任务…");
    try {
      const parsedPcaDimension = parseIntegerParameter(
        "PCA Dimension",
        pcaDimension,
        2,
        1024,
      );
      const parsedMinSamples = parseIntegerParameter(
        "Min Samples",
        minSamples,
        1,
        10_000,
      );
      const parsedMinClusterSize = parseIntegerParameter(
        "Min Cluster Size",
        minClusterSize,
        2,
        10_000,
      );
      const submitted = await apiFetch<{ cluster_run_id: string }>(
        "/api/v1/cluster-runs",
        {
          method: "POST",
          body: JSON.stringify({
            workspace_id: workspaceId,
            embedding_type: embeddingType,
            pca_dimension: parsedPcaDimension,
            min_samples: parsedMinSamples,
            min_cluster_size: parsedMinClusterSize,
          }),
        },
      );
      setSelectedRunId(submitted.cluster_run_id);
      const submittedRun = await apiFetch<ClusterRun>(
        `/api/v1/cluster-runs/${encodeURIComponent(submitted.cluster_run_id)}?workspace_id=${encodeURIComponent(workspaceId)}`,
      );
      setRuns((current) => [
        submittedRun,
        ...current.filter(
          (run) => run.cluster_run_id !== submittedRun.cluster_run_id,
        ),
      ]);
      setRunNotice("全量重聚类已提交，完成后会自动展示结果…");
      await loadAssetStatus();
    } catch (requestError) {
      setRunNotice(null);
      setError(
        requestError instanceof Error
          ? requestError.message
          : "全量重聚类启动失败",
      );
    } finally {
      setRunning(false);
    }
  };

  const renameCurrentCluster = async (value: string) => {
    if (!selectedCurrentCluster || !value.trim()) return;
    setCurrentMutation("rename");
    setCurrentError(null);
    setCurrentNotice(null);
    try {
      const params = new URLSearchParams({ workspace_id: workspaceId });
      const updated = await apiFetch<CurrentCluster>(
        `/api/v1/clusters/${encodeURIComponent(selectedCurrentCluster.cluster_id)}?${params}`,
        {
          method: "PATCH",
          body: JSON.stringify({ name: value.trim() }),
        },
      );
      setCurrentClusters((current) =>
        current.map((cluster) =>
          cluster.cluster_id === updated.cluster_id ? updated : cluster,
        ),
      );
      setCurrentNotice(`已重命名为 ${updated.name}`);
    } catch (requestError) {
      setCurrentError(
        requestError instanceof Error ? requestError.message : "簇重命名失败",
      );
    } finally {
      setCurrentMutation("");
    }
  };

  const updateCurrentMode = async (mode: CurrentClusterMode) => {
    if (!selectedCurrentCluster || selectedCurrentCluster.mode === mode) return;
    setCurrentMutation("mode");
    setCurrentError(null);
    setCurrentNotice(null);
    try {
      const params = new URLSearchParams({ workspace_id: workspaceId });
      const updated = await apiFetch<CurrentCluster>(
        `/api/v1/clusters/${encodeURIComponent(selectedCurrentCluster.cluster_id)}?${params}`,
        {
          method: "PATCH",
          body: JSON.stringify({ mode }),
        },
      );
      setCurrentClusters((current) =>
        current.map((cluster) =>
          cluster.cluster_id === updated.cluster_id ? updated : cluster,
        ),
      );
      setCurrentNotice(`已切换为${currentClusterModeLabel(updated.mode)}`);
    } catch (requestError) {
      setCurrentError(
        requestError instanceof Error ? requestError.message : "簇模式更新失败",
      );
    } finally {
      setCurrentMutation("");
    }
  };

  const attachCurrentMembers = async () => {
    if (!selectedCurrentCluster) return;
    const assetIds = parseAssetIds(assetIdsInput);
    if (!assetIds.length) {
      setCurrentError("请输入至少一个 Asset ID");
      return;
    }
    setCurrentMutation("attach");
    setCurrentError(null);
    setCurrentNotice(null);
    try {
      const params = new URLSearchParams({ workspace_id: workspaceId });
      await apiFetch<{ cluster_id: string; asset_ids: string[] }>(
        `/api/v1/clusters/${encodeURIComponent(selectedCurrentCluster.cluster_id)}/members:attach?${params}`,
        {
          method: "POST",
          body: JSON.stringify({ asset_ids: assetIds }),
        },
      );
      setAssetIdsInput("");
      setCurrentNotice(`已加入 ${assetIds.length} 个 Asset`);
      await loadCurrentMembers();
    } catch (requestError) {
      setCurrentError(
        requestError instanceof Error ? requestError.message : "成员加入失败",
      );
    } finally {
      setCurrentMutation("");
    }
  };

  const detachCurrentMember = async (assetId: string) => {
    if (!selectedCurrentCluster) return;
    setCurrentMutation(`detach:${assetId}`);
    setCurrentError(null);
    setCurrentNotice(null);
    try {
      const params = new URLSearchParams({ workspace_id: workspaceId });
      await apiFetch<{ cluster_id: string; asset_ids: string[] }>(
        `/api/v1/clusters/${encodeURIComponent(selectedCurrentCluster.cluster_id)}/members:detach?${params}`,
        {
          method: "POST",
          body: JSON.stringify({ asset_ids: [assetId] }),
        },
      );
      setCurrentMembers((current) =>
        current.filter((member) => member.asset_id !== assetId),
      );
      setCurrentNotice(`已移出 Asset ${assetId}`);
    } catch (requestError) {
      setCurrentError(
        requestError instanceof Error ? requestError.message : "成员移出失败",
      );
    } finally {
      setCurrentMutation("");
    }
  };

  const moveCurrentMember = async (assetId: string) => {
    const targetClusterId = moveTargets[assetId];
    const targetCluster = currentClusters.find(
      (cluster) => cluster.cluster_id === targetClusterId,
    );
    if (!targetCluster) {
      setCurrentError("请选择目标簇");
      return;
    }
    setCurrentMutation(`move:${assetId}`);
    setCurrentError(null);
    setCurrentNotice(null);
    try {
      const params = new URLSearchParams({ workspace_id: workspaceId });
      await apiFetch<{ cluster_id: string; asset_ids: string[] }>(
        `/api/v1/clusters/${encodeURIComponent(targetCluster.cluster_id)}/members:attach?${params}`,
        {
          method: "POST",
          body: JSON.stringify({ asset_ids: [assetId] }),
        },
      );
      setCurrentMembers((current) =>
        current.filter((member) => member.asset_id !== assetId),
      );
      setMoveTargets((current) => {
        const next = { ...current };
        delete next[assetId];
        return next;
      });
      setCurrentNotice(`已将 Asset ${assetId} 移动到 ${targetCluster.name}`);
    } catch (requestError) {
      setCurrentError(
        requestError instanceof Error ? requestError.message : "成员移动失败",
      );
    } finally {
      setCurrentMutation("");
    }
  };

  const graphLayout = useMemo(() => {
    const columnCount = Math.max(
      1,
      Math.ceil(Math.sqrt(Math.max(capsules.length, 1) * 1.25)),
    );
    const rowCount = Math.max(1, Math.ceil(capsules.length / columnCount));
    const cellWidth = 340;
    const cellHeight = 320;
    const width = Math.max(760, columnCount * cellWidth + 180);
    const height = Math.max(520, rowCount * cellHeight + 160);
    const offsetX = (width - columnCount * cellWidth) / 2;
    const offsetY = (height - rowCount * cellHeight) / 2;
    const groups = capsules.map((capsule, index) => {
      const radius = clamp(
        106 + Math.sqrt(Math.max(capsule.member_count, 1)) * 7,
        112,
        148,
      );
      const members = membersByCapsule[capsule.cluster_capsule_id] || [];
      return {
        capsule,
        color: GROUP_COLORS[index % GROUP_COLORS.length],
        diameter: radius * 2,
        left:
          offsetX +
          (index % columnCount) * cellWidth +
          cellWidth / 2 -
          radius,
        top:
          offsetY +
          Math.floor(index / columnCount) * cellHeight +
          cellHeight / 2 -
          radius,
        members,
        memberNodes: placeMemberNodes(members, radius),
      };
    });
    return { width, height, groups };
  },
    [capsules, membersByCapsule],
  );

  return (
    <DemoShell
      active="clusters"
      workspaceControl={
        <WorkspaceSelect
          workspaceId={workspaceId}
          workspaces={workspaces}
          loading={workspacesLoading}
          onChange={(nextWorkspaceId) => {
            runRequestRef.current += 1;
            currentClusterRequestRef.current += 1;
            assetStatusRequestRef.current += 1;
            currentMemberRequestRef.current += 1;
            setRuns([]);
            setSelectedRunId("");
            setCapsules([]);
            setMembersByCapsule({});
            setCurrentClusters([]);
            setCurrentMembers([]);
            setAssetStatus(null);
            setError(null);
            setWorkspaceId(nextWorkspaceId);
          }}
        />
      }
      eyebrow="CLUSTER LAB / LIVE"
      title="从相似中，看见结构。"
      description="首次达到样本阈值会自动初始化；之后新增 Asset 只做增量归簇，全量重聚类由用户手动启动。"
      actions={
        <button
          className="primary-action"
          disabled={running || Boolean(activeRunForDimension)}
          title="只有点击此按钮，才会对当前维度执行全量重聚类"
          onClick={createRun}
        >
          {running
            ? "正在提交全量重聚类…"
            : activeRunForDimension
              ? "正在全量重聚类…"
              : "＋ 全量重聚类当前维度"}
        </button>
      }
    >
      <section className="run-selector">
        <div>
          <label>
            Feature Type
            <select
              value={embeddingType}
              onChange={(event) => {
                const nextEmbeddingType = event.target.value;
                setEmbeddingType(nextEmbeddingType);
                setSelectedRunId(
                  runs.find(
                    (run) => run.embedding_type === nextEmbeddingType,
                  )?.cluster_run_id || "",
                );
                setRunNotice(null);
              }}
            >
              {FEATURE_TYPES.map((item) => (
                <option value={item.value} key={item.value}>
                  {item.label}（{item.value}）
                </option>
              ))}
            </select>
          </label>
          <label>
            Cluster Run
            <select
              value={selectedRunId}
              onChange={(event) => setSelectedRunId(event.target.value)}
            >
              {!runs.length && <option value="">尚无 Run</option>}
              {runs.map((run) => (
                <option value={run.cluster_run_id} key={run.cluster_run_id}>
                  {runOptionLabel(run)}
                </option>
              ))}
            </select>
          </label>
          <label>
            PCA Dimension
            <input
              type="number"
              min="2"
              max="1024"
              step="1"
              value={pcaDimension}
              onChange={(event) => setPcaDimension(event.target.value)}
            />
          </label>
          <label>
            Min Samples
            <input
              type="number"
              min="1"
              max="10000"
              step="1"
              value={minSamples}
              onChange={(event) => setMinSamples(event.target.value)}
            />
          </label>
          <label>
            Min Cluster Size
            <input
              type="number"
              min="2"
              max="10000"
              step="1"
              value={minClusterSize}
              onChange={(event) => setMinClusterSize(event.target.value)}
            />
          </label>
        </div>
        <div className="run-parameters">
          <span>
            <small>PCA</small>
            <strong>{runPcaDimension(selectedRun)}</strong>
          </span>
          <span>
            <small>MIN SAMPLES</small>
            <strong>
              {runParameter(selectedRun?.parameters.min_samples)}
            </strong>
          </span>
          <span>
            <small>MIN CLUSTER</small>
            <strong>
              {runParameter(selectedRun?.parameters.min_cluster_size)}
            </strong>
          </span>
          <span>
            <small>SAMPLES</small>
            <strong>{selectedRun?.sample_count ?? 0}</strong>
          </span>
          <span>
            <small>CLUSTERS</small>
            <strong>{selectedRun?.cluster_count ?? 0}</strong>
          </span>
          <span>
            <small>NOISE RATIO</small>
            <strong>
              {selectedRun?.noise_ratio == null
                ? "—"
                : `${(selectedRun.noise_ratio * 100).toFixed(1)}%`}
            </strong>
          </span>
          <span>
            <small>STATUS</small>
            <StatusBadge status={selectedRun?.status || "pending"} />
          </span>
        </div>
        <p
          className={`full-recluster-note ${runNotice ? "active" : ""}`}
          role={runNotice ? "status" : undefined}
          aria-live="polite"
        >
          {runNotice ||
            "历史 Cluster Run 仅供查看；只有点击“全量重聚类当前维度”才会重新计算全部动态样本。"}
        </p>
      </section>

      <section className="cluster-asset-status">
        <header>
          <div>
            <span className="eyebrow">INCREMENTAL STATUS</span>
            <h2>Asset 聚类进度</h2>
            <p>
              首次达到
              {assetStatus?.bootstrap_minimum_count == null
                ? "配置的"
                : ` ${assetStatus.bootstrap_minimum_count} 条`}
              样本阈值会自动初始化；初始化后新增 Asset 只增量归簇，不会自动触发全量重聚类。
            </p>
          </div>
          <span
            className={`asset-status-phase ${assetStatus?.initialized ? "initialized" : "waiting"}`}
          >
            {assetStatusLoading
              ? "同步中…"
              : assetStatus?.initialized
                ? "增量运行中"
                : "等待首次初始化"}
          </span>
        </header>

        <div className="asset-status-summary">
          <span>
            <small>基线样本数</small>
            <strong>{assetStatus?.baseline_sample_count ?? "—"}</strong>
            <em>{assetStatus?.baseline_cluster_run_id || "尚无基线 Run"}</em>
          </span>
          <span>
            <small>当前 eligible</small>
            <strong>{assetStatus?.eligible_asset_count ?? "—"}</strong>
            <em>当前维度可参与样本</em>
          </span>
          <span>
            <small>新增 Asset</small>
            <strong>{assetStatus?.new_asset_count ?? "—"}</strong>
            <em>相对基线新增</em>
          </span>
          <span>
            <small>已增量归簇</small>
            <strong>{assetStatus?.incrementally_clustered_count ?? "—"}</strong>
            <em>自动加入动态 / 开放常驻簇</em>
          </span>
          <span>
            <small>待聚类</small>
            <strong>{assetStatus?.pending_count ?? "—"}</strong>
            <em>等待下次增量处理</em>
          </span>
          <span>
            <small>手动管理</small>
            <strong>{assetStatus?.manual_management_count ?? "—"}</strong>
            <em>不由算法自动调整</em>
          </span>
        </div>

        <div className="asset-status-table">
          <div className="asset-status-row asset-status-head">
            <span>新增 Asset</span>
            <span>状态</span>
            <span>目标簇</span>
            <span>分数</span>
          </div>
          {assetStatusError && (
            <p className="asset-status-message error">{assetStatusError}</p>
          )}
          {!assetStatusError && assetStatusLoading && !assetStatus && (
            <p className="asset-status-message">正在读取 Asset 聚类状态…</p>
          )}
          {!assetStatusError && !assetStatusLoading && !assetStatus?.items.length && (
            <p className="asset-status-message">当前维度暂无新增 Asset。</p>
          )}
          {assetStatus?.items.map((item) => (
            <div className="asset-status-row" key={item.asset_id}>
              <span className="asset-status-identity">
                <strong>{item.asset_name || item.file_name || item.asset_id}</strong>
                <small title={item.asset_id}>{item.asset_id}</small>
                <em>{item.asset_type}</em>
              </span>
              <span>
                <strong className={`asset-status-label ${item.status}`}>
                  {clusterAssetStatusLabel(item.status)}
                </strong>
                <small>{item.member_source || "尚无成员来源"}</small>
              </span>
              <span>
                <strong>{item.cluster_name || item.cluster_id || "尚未分配"}</strong>
                <small>
                  {item.cluster_mode
                    ? currentClusterModeLabel(item.cluster_mode)
                    : "—"}
                </small>
              </span>
              <strong className="asset-status-score">
                {item.score == null ? "—" : item.score.toFixed(3)}
              </strong>
            </div>
          ))}
        </div>
      </section>

      <section
        className="current-cluster-workspace"
        id="current-clusters"
        ref={currentClusterWorkspaceRef}
      >
        <header>
          <div>
            <span className="eyebrow">CURRENT CLUSTERS</span>
            <h2>当前聚类</h2>
            <p>
              当前维度为
              {FEATURE_TYPES.find((item) => item.value === embeddingType)
                ?.label || embeddingType}
              。新增 Asset 日常只走增量；仅点击页面顶部按钮才执行全量重聚类。
            </p>
          </div>
          <button
            type="button"
            className="secondary-action"
            disabled={currentLoading || assetStatusLoading}
            onClick={() => void refreshClusterDimension()}
          >
            {currentLoading || assetStatusLoading ? "加载中…" : "刷新状态"}
          </button>
        </header>

        <div className="current-cluster-layout">
          <nav aria-label="当前簇列表">
            {!currentClusters.length && (
              <p>{currentLoading ? "正在读取当前簇…" : "当前维度尚无簇"}</p>
            )}
            {currentClusters.map((cluster, index) => (
              <button
                type="button"
                className={
                  selectedCurrentCluster?.cluster_id === cluster.cluster_id
                    ? "active"
                    : ""
                }
                aria-pressed={
                  selectedCurrentCluster?.cluster_id === cluster.cluster_id
                }
                onClick={() => setSelectedCurrentClusterId(cluster.cluster_id)}
                key={cluster.cluster_id}
              >
                <span>{String(index + 1).padStart(2, "0")}</span>
                <strong>{cluster.name}</strong>
                <small className={`cluster-mode-badge ${cluster.mode}`}>
                  {currentClusterModeLabel(cluster.mode)}
                </small>
              </button>
            ))}
          </nav>

          <div className="current-cluster-detail">
            {!selectedCurrentCluster ? (
              <div className="current-cluster-empty">
                {currentError || "选择一个当前簇进行维护"}
              </div>
            ) : (
              <>
                <div className="current-cluster-heading">
                  <div>
                    <small>{selectedCurrentCluster.embedding_type}</small>
                    <h3>{selectedCurrentCluster.name}</h3>
                    <p>
                      {selectedCurrentCluster.description || "暂无簇描述"}
                    </p>
                  </div>
                  <div className="current-cluster-heading-actions">
                    <strong>{currentMembers.length} MEMBERS</strong>
                    <button
                      type="button"
                      disabled={Boolean(currentMutation)}
                      onClick={() => {
                        const name = window.prompt(
                          "输入新的簇名称",
                          selectedCurrentCluster.name,
                        );
                        if (name) void renameCurrentCluster(name);
                      }}
                    >
                      {currentMutation === "rename" ? "重命名中…" : "重命名"}
                    </button>
                  </div>
                </div>

                <div className="cluster-mode-controls" role="group" aria-label="簇模式">
                  {CURRENT_CLUSTER_MODES.map((mode) => (
                    <button
                      type="button"
                      className={
                        selectedCurrentCluster.mode === mode.value
                          ? "active"
                          : ""
                      }
                      disabled={Boolean(currentMutation)}
                      title={mode.help}
                      onClick={() => void updateCurrentMode(mode.value)}
                      key={mode.value}
                    >
                      <strong>{mode.label}</strong>
                      <small>{mode.help}</small>
                    </button>
                  ))}
                </div>

                {selectedCurrentCluster.mode !== "dynamic" && (
                  <div className="current-cluster-attach">
                    <label htmlFor="resident-asset-ids">
                      手动加入 Asset
                      <input
                        id="resident-asset-ids"
                        value={assetIdsInput}
                        placeholder="输入 Asset ID，使用逗号、空格或换行分隔"
                        onChange={(event) => setAssetIdsInput(event.target.value)}
                      />
                    </label>
                    <button
                      type="button"
                      className="primary-action"
                      disabled={Boolean(currentMutation)}
                      onClick={() => void attachCurrentMembers()}
                    >
                      {currentMutation === "attach" ? "正在加入…" : "批量加入"}
                    </button>
                  </div>
                )}

                {(currentError || currentNotice) && (
                  <div
                    className={`current-cluster-message ${currentError ? "error" : "success"}`}
                  >
                    {currentError || currentNotice}
                  </div>
                )}

                <p className="current-member-help">
                  每个 Asset 的操作在表格最右侧。移动目标只显示常驻簇；若没有目标，先在左侧选择另一个簇并设为开放常驻或手动管理。
                </p>

                <div className="current-member-table">
                  <div className="current-member-row current-member-head">
                    <span>缩略图</span>
                    <span>Asset ID</span>
                    <span>来源</span>
                    <span>归类分数</span>
                    <span>移动 / 移出</span>
                  </div>
                  {!currentMembers.length && (
                    <p>这个簇当前没有成员。</p>
                  )}
                  {currentMembers.map((member) => (
                    <div className="current-member-row" key={member.asset_id}>
                      <CurrentMemberThumbnail
                        assetId={member.asset_id}
                        workspaceId={workspaceId}
                      />
                      <strong title={member.asset_id}>{member.asset_id}</strong>
                      <span>{member.source}</span>
                      <span>
                        {member.score == null ? "人工指定" : member.score.toFixed(3)}
                      </span>
                      <span className="current-member-actions">
                        <select
                          aria-label={`为 ${member.asset_id} 选择目标簇`}
                          title="移动到其他簇"
                          value={moveTargets[member.asset_id] || ""}
                          disabled={Boolean(currentMutation)}
                          onChange={(event) =>
                            setMoveTargets((current) => ({
                              ...current,
                              [member.asset_id]: event.target.value,
                            }))
                          }
                        >
                          <option value="">
                            {availableMoveTargets.length
                              ? "选择目标常驻簇"
                              : "暂无常驻目标簇"}
                          </option>
                          {availableMoveTargets.map((cluster) => (
                              <option
                                value={cluster.cluster_id}
                                key={cluster.cluster_id}
                              >
                                {cluster.name}
                              </option>
                            ))}
                        </select>
                        <button
                          type="button"
                          className="move-button"
                          disabled={
                            Boolean(currentMutation) ||
                            !moveTargets[member.asset_id]
                          }
                          onClick={() => void moveCurrentMember(member.asset_id)}
                        >
                          {currentMutation === `move:${member.asset_id}`
                            ? "移动中…"
                            : "移动"}
                        </button>
                        <button
                          type="button"
                          disabled={Boolean(currentMutation)}
                          onClick={() => void detachCurrentMember(member.asset_id)}
                        >
                          {currentMutation === `detach:${member.asset_id}`
                            ? "移出中…"
                            : "移出"}
                        </button>
                      </span>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>
      </section>

      {error && (
        <div className="asset-empty">
          <strong>聚类服务返回错误</strong>
          <span>{error}</span>
        </div>
      )}

      <div className="cluster-overview">
        <section className="cluster-map">
          <header>
            <div>
              <span className="eyebrow">SEMANTIC GROUPS</span>
              <h2>聚类知识图谱</h2>
            </div>
            <small>每种颜色代表一个真实 Cluster Capsule</small>
          </header>
          <ClusterKnowledgeGraph
            layout={graphLayout}
            selectedCapsuleId={selectedCapsule?.cluster_capsule_id || ""}
            emptyMessage={
              isActiveRun(selectedRun)
                ? "正在计算聚类…"
                : selectedRun
                  ? "本次聚类未生成语义分组"
                  : "选择 Embedding Type 并创建第一个 Run"
            }
            onSelect={setSelectedCapsuleId}
          />
          <footer>
            {graphLayout.groups.map((group) => (
              <span key={group.capsule.cluster_capsule_id}>
                <i style={{ background: group.color }} />
                {group.capsule.effective_name}
              </span>
            ))}
          </footer>
        </section>

        <section className="cluster-cards">
          <header>
            <div>
              <span className="eyebrow">CLUSTER CAPSULES</span>
              <h2>语义分组</h2>
            </div>
            <div className="cluster-card-header-actions">
              <strong>{capsules.length}</strong>
              <button
                type="button"
                onClick={() =>
                  currentClusterWorkspaceRef.current?.scrollIntoView({
                    behavior: "smooth",
                    block: "start",
                  })
                }
              >
                管理当前簇
              </button>
            </div>
          </header>
          <div className="cluster-card-list">
            {capsules.map((capsule, index) => (
              <article
                className={
                  selectedCapsule?.cluster_capsule_id ===
                  capsule.cluster_capsule_id
                    ? "active"
                    : ""
                }
                onClick={() => setSelectedCapsuleId(capsule.cluster_capsule_id)}
                ref={(node) => {
                  if (node) {
                    capsuleCardRefs.current.set(
                      capsule.cluster_capsule_id,
                      node,
                    );
                  } else {
                    capsuleCardRefs.current.delete(
                      capsule.cluster_capsule_id,
                    );
                  }
                }}
                key={capsule.cluster_capsule_id}
              >
                <div
                  className="cluster-number"
                  style={{
                    background: GROUP_COLORS[index % GROUP_COLORS.length],
                  }}
                >
                  {String(index + 1).padStart(2, "0")}
                </div>
                <div>
                  <small>{capsule.embedding_type}</small>
                  <h3>{capsule.effective_name}</h3>
                  <p>{capsule.effective_description}</p>
                  <footer>
                    <span>{capsule.member_count} MEMBERS</span>
                    <span>
                      {capsule.average_membership_probability.toFixed(2)} AVG
                      PROB.
                    </span>
                  </footer>
                </div>
              </article>
            ))}
          </div>
        </section>
      </div>

      {selectedCapsule && (
        <section className="cluster-detail" ref={clusterDetailRef}>
          <header>
            <div>
              <span className="eyebrow">SELECTED CAPSULE</span>
              <h2>{selectedCapsule.effective_name}</h2>
              <p>{selectedCapsule.effective_description}</p>
            </div>
            <div>
              <StatusBadge status="completed" />
            </div>
          </header>
          <div className="cluster-member-summary">
            <span>
              已加载 <strong>{selectedMembers.length}</strong> /{" "}
              {selectedCapsule.member_count} 个 Assets
            </span>
            {selectedMembers.length !== selectedCapsule.member_count && (
              <small>成员数据仍在加载，请稍候…</small>
            )}
          </div>
          <div className="representative-strip">
            {selectedMembers.map((asset, index) => (
              <Link
                className="representative-asset-link"
                href={`/assets/${encodeURIComponent(asset.asset_id)}`}
                aria-label={`打开 Asset 详情：${asset.asset_name || asset.file_name}`}
                key={asset.asset_id}
              >
                <article>
                  <AssetThumb
                    preview={asset.preview_url}
                    name={asset.asset_name || asset.file_name}
                    type={asset.asset_type}
                  />
                  <span>
                    {asset.asset_id === selectedCapsule.medoid_asset_id
                      ? "MEDOID"
                      : `MEMBER ${index + 1}`}
                  </span>
                  <strong>{asset.asset_name || asset.file_name}</strong>
                  <small className="representative-asset-id" title={asset.asset_id}>
                    {asset.asset_id}
                  </small>
                </article>
              </Link>
            ))}
          </div>
        </section>
      )}
    </DemoShell>
  );
}
