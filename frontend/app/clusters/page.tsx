"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
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

const EMBEDDING_TYPES = [
  "native_multimodal",
  "asset_description",
  "subject_content",
  "scene_theme",
  "visual_style",
  "color_composition",
  "mood_atmosphere",
];

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

export default function ClustersPage() {
  const [runs, setRuns] = useState<ClusterRun[]>([]);
  const [selectedRunId, setSelectedRunId] = useState("");
  const [embeddingType, setEmbeddingType] = useState("native_multimodal");
  const [capsules, setCapsules] = useState<ClusterCapsule[]>([]);
  const [selectedCapsuleId, setSelectedCapsuleId] = useState("");
  const [membersByCapsule, setMembersByCapsule] = useState<
    Record<string, ClusterMember[]>
  >({});
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadRuns = useCallback(async () => {
    try {
      const payload = await apiFetch<{ items: ClusterRun[] }>(
        `/api/v1/cluster-runs?workspace_id=${WORKSPACE_ID}&limit=100`,
      );
      setRuns(payload.items);
      setSelectedRunId((current) =>
        current && payload.items.some((run) => run.cluster_run_id === current)
          ? current
          : payload.items[0]?.cluster_run_id || "",
      );
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
          setSelectedCapsuleId((current) =>
            current &&
            payload.items.some(
              (capsule) => capsule.cluster_capsule_id === current,
            )
              ? current
              : payload.items[0]?.cluster_capsule_id || "",
          );
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

  const createRun = async () => {
    setRunning(true);
    setError(null);
    try {
      const submitted = await apiFetch<{ cluster_run_id: string }>(
        "/api/v1/cluster-runs",
        {
          method: "POST",
          body: JSON.stringify({
            workspace_id: WORKSPACE_ID,
            embedding_type: embeddingType,
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

  const groupLayout = useMemo(
    () =>
      capsules.map((capsule, index) => {
        const column = index % 3;
        const row = Math.floor(index / 3);
        return {
          capsule,
          left: 7 + column * 31,
          top: 7 + row * 43,
          width: 25,
          height: 34,
          color: GROUP_COLORS[index % GROUP_COLORS.length],
          members: membersByCapsule[capsule.cluster_capsule_id] || [],
        };
      }),
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
            Embedding Type
            <select
              value={embeddingType}
              onChange={(event) => setEmbeddingType(event.target.value)}
            >
              {EMBEDDING_TYPES.map((item) => (
                <option value={item} key={item}>
                  {item}
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
                  {run.embedding_type} · {run.status} · {run.sample_count}
                </option>
              ))}
            </select>
          </label>
        </div>
        <div className="run-parameters">
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
          <div className="scatter-plot cluster-bubble-map" aria-label="聚类知识图谱">
            {groupLayout.map((group) => (
              <button
                className={`cluster-bubble-group ${
                  selectedCapsule?.cluster_capsule_id ===
                  group.capsule.cluster_capsule_id
                    ? "active"
                    : ""
                }`}
                style={{
                  left: `${group.left}%`,
                  top: `${group.top}%`,
                  width: `${group.width}%`,
                  height: `${group.height}%`,
                  borderColor: group.color,
                  backgroundColor: `${group.color}18`,
                }}
                onClick={() =>
                  setSelectedCapsuleId(group.capsule.cluster_capsule_id)
                }
                key={group.capsule.cluster_capsule_id}
              >
                <strong>{group.capsule.effective_name}</strong>
                <small>{group.capsule.member_count} Assets</small>
                {group.members.slice(0, 12).map((member, index) => (
                  <i
                    style={{
                      left: `${12 + ((index * 29) % 76)}%`,
                      top: `${34 + ((index * 37) % 54)}%`,
                      backgroundColor: group.color,
                    }}
                    title={member.asset_name || member.file_name}
                    key={member.asset_id}
                  />
                ))}
              </button>
            ))}
            {!groupLayout.length && (
              <div className="cluster-map-empty">
                {selectedRun?.status === "running"
                  ? "正在计算聚类…"
                  : "选择 Embedding Type 并创建第一个 Run"}
              </div>
            )}
          </div>
          <footer>
            {groupLayout.map((group) => (
              <span key={group.capsule.cluster_capsule_id}>
                <i style={{ background: group.color }} />
                {group.capsule.effective_name}
              </span>
            ))}
          </footer>
        </section>

        <section className="cluster-cards">
          <header>
            <span className="eyebrow">CLUSTER CAPSULES</span>
            <h2>{capsules.length} 个语义分组</h2>
          </header>
          {capsules.map((capsule, index) => (
            <article
              className={
                selectedCapsule?.cluster_capsule_id ===
                capsule.cluster_capsule_id
                  ? "active"
                  : ""
              }
              onClick={() => setSelectedCapsuleId(capsule.cluster_capsule_id)}
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
                    {capsule.average_membership_probability.toFixed(2)} AVG PROB.
                  </span>
                </footer>
              </div>
            </article>
          ))}
        </section>
      </div>

      {selectedCapsule && (
        <section className="cluster-detail">
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
          <div className="representative-strip">
            {selectedMembers.slice(0, 6).map((asset, index) => (
              <article key={asset.asset_id}>
                <AssetThumb
                  preview={asset.preview_url}
                  name={asset.asset_name || asset.file_name}
                  type={asset.asset_type}
                />
                <span>{index === 0 ? "MEDOID" : `MEMBER ${index + 1}`}</span>
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
