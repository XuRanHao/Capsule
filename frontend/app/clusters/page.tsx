"use client";

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
  type ClusterCapsule,
  type ClusterMember,
  type ClusterRun,
  WORKSPACE_ID,
  apiFetch,
} from "../lib/api";

const FEATURE_TYPES = [
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
  const [runs, setRuns] = useState<ClusterRun[]>([]);
  const [selectedRunId, setSelectedRunId] = useState("");
  const [embeddingType, setEmbeddingType] = useState<string>(
    FEATURE_TYPES[0].value,
  );
  const [pcaDimension, setPcaDimension] = useState("8");
  const [minSamples, setMinSamples] = useState("1");
  const [minClusterSize, setMinClusterSize] = useState("3");
  const [capsules, setCapsules] = useState<ClusterCapsule[]>([]);
  const [selectedCapsuleId, setSelectedCapsuleId] = useState("");
  const [membersByCapsule, setMembersByCapsule] = useState<
    Record<string, ClusterMember[]>
  >({});
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
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
    try {
      const payload = await apiFetch<{ items: ClusterRun[] }>(
        `/api/v1/cluster-runs?workspace_id=${WORKSPACE_ID}&limit=100`,
      );
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
      setError(
        requestError instanceof Error ? requestError.message : "聚类记录加载失败",
      );
    }
  }, []);

  useEffect(() => {
    const initial = window.setTimeout(() => void loadRuns(), 0);
    const polling = window.setInterval(() => void loadRuns(), 2500);
    return () => {
      window.clearTimeout(initial);
      window.clearInterval(polling);
    };
  }, [loadRuns]);

  const selectedRun =
    runs.find((run) => run.cluster_run_id === selectedRunId) ?? runs[0];

  useEffect(() => {
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
        `/api/v1/cluster-runs/${selectedRun.cluster_run_id}/capsules?workspace_id=${WORKSPACE_ID}`,
      )
        .then(async (payload) => {
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
                `/api/v1/cluster-capsules/${capsule.cluster_capsule_id}/members?workspace_id=${WORKSPACE_ID}`,
              );
              return [capsule.cluster_capsule_id, members.items] as const;
            }),
          );
          setMembersByCapsule(Object.fromEntries(entries));
        })
        .catch((requestError: unknown) =>
          setError(
            requestError instanceof Error
              ? requestError.message
              : "Cluster Capsule 加载失败",
          ),
        );
    }, 0);
    return () => window.clearTimeout(timer);
  }, [selectedRun]);

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
    setRunning(true);
    setError(null);
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
            workspace_id: WORKSPACE_ID,
            embedding_type: embeddingType,
            pca_dimension: parsedPcaDimension,
            min_samples: parsedMinSamples,
            min_cluster_size: parsedMinClusterSize,
          }),
        },
      );
      setSelectedRunId(submitted.cluster_run_id);
      await loadRuns();
    } catch (requestError) {
      setError(
        requestError instanceof Error ? requestError.message : "创建聚类失败",
      );
    } finally {
      setRunning(false);
    }
  };

  const updateName = async (value: string) => {
    if (!selectedCapsule || !value.trim()) return;
    const updated = await apiFetch<ClusterCapsule>(
      `/api/v1/cluster-capsules/${selectedCapsule.cluster_capsule_id}?workspace_id=${WORKSPACE_ID}`,
      {
        method: "PATCH",
        body: JSON.stringify({ name: value.trim() }),
      },
    );
    setCapsules((current) =>
      current.map((capsule) =>
        capsule.cluster_capsule_id === updated.cluster_capsule_id
          ? updated
          : capsule,
      ),
    );
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
      eyebrow="CLUSTER LAB / LIVE"
      title="从相似中，看见结构。"
      description="Cluster Run、Capsule 和成员全部来自 PostgreSQL 与 Milvus，不再使用固定散点。"
      actions={
        <button className="primary-action" disabled={running} onClick={createRun}>
          {running ? "正在创建 Run…" : "＋ 创建 Cluster Run"}
        </button>
      }
    >
      <section className="run-selector">
        <div>
          <label>
            Feature Type
            <select
              value={embeddingType}
              onChange={(event) => setEmbeddingType(event.target.value)}
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
              selectedRun?.status === "running"
                ? "正在计算聚类…"
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
            <strong>{capsules.length}</strong>
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
              <button
                onClick={() => {
                  const name = window.prompt(
                    "输入新的 Cluster 名称",
                    selectedCapsule.effective_name,
                  );
                  if (name) void updateName(name);
                }}
              >
                用户改名
              </button>
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
              <article key={asset.asset_id}>
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
              </article>
            ))}
          </div>
          <div className="member-table">
            <div className="data-row member-row data-head">
              <span>成员</span>
              <span>类型</span>
              <span>Membership Probability</span>
              <span>来源</span>
            </div>
            {selectedMembers.map((asset) => (
              <div className="data-row member-row" key={asset.asset_id}>
                <strong>{asset.asset_name || asset.file_name}</strong>
                <span>{asset.asset_type}</span>
                <span className="membership-meter">
                  <i
                    style={{ width: `${asset.membership_probability * 100}%` }}
                  />
                  <b>{asset.membership_probability.toFixed(2)}</b>
                </span>
                <span>{asset.relative_path}</span>
              </div>
            ))}
          </div>
        </section>
      )}
    </DemoShell>
  );
}
