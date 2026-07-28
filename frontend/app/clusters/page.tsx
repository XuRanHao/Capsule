"use client";

import { useMemo, useState } from "react";
import DemoShell, {
  AssetThumb,
  StatusBadge,
} from "../components/DemoShell";
import { DEMO_ASSETS, DEMO_CLUSTERS } from "../lib/demo-data";

const RUNS = [
  {
    id: "run_native_20260728",
    type: "native_multimodal",
    created: "今天 15:28",
    sampleCount: 126,
    clusters: 8,
    noise: 0.087,
    minClusterSize: 5,
    minSamples: 3,
    pca: 64,
  },
  {
    id: "run_mood_20260727",
    type: "mood_atmosphere",
    created: "昨天 18:06",
    sampleCount: 119,
    clusters: 6,
    noise: 0.126,
    minClusterSize: 5,
    minSamples: 3,
    pca: 64,
  },
  {
    id: "run_style_20260726",
    type: "visual_style",
    created: "7 月 26 日",
    sampleCount: 114,
    clusters: 7,
    noise: 0.096,
    minClusterSize: 5,
    minSamples: 3,
    pca: 64,
  },
];

const DOTS = [
  [12, 22, 0],
  [17, 31, 0],
  [22, 18, 0],
  [27, 27, 0],
  [31, 15, 0],
  [48, 68, 1],
  [53, 74, 1],
  [59, 65, 1],
  [64, 72, 1],
  [69, 61, 1],
  [72, 25, 2],
  [77, 33, 2],
  [82, 21, 2],
  [87, 29, 2],
  [40, 38, 3],
  [45, 44, 3],
  [50, 35, 3],
  [54, 42, 3],
  [33, 79, -1],
  [91, 82, -1],
  [8, 65, -1],
];

export default function ClustersPage() {
  const [selectedRunId, setSelectedRunId] = useState(RUNS[0].id);
  const [selectedClusterId, setSelectedClusterId] = useState(
    DEMO_CLUSTERS[0].id,
  );
  const [clusters, setClusters] = useState(DEMO_CLUSTERS);
  const [renaming, setRenaming] = useState(false);
  const [running, setRunning] = useState(false);
  const selectedRun = RUNS.find((run) => run.id === selectedRunId) ?? RUNS[0];
  const selectedCluster =
    clusters.find((cluster) => cluster.id === selectedClusterId) ?? clusters[0];
  const members = useMemo(
    () =>
      selectedCluster.assetIds
        .map((id) => DEMO_ASSETS.find((asset) => asset.id === id))
        .filter((asset) => asset !== undefined),
    [selectedCluster],
  );

  const toggleFavorite = (id: string) => {
    setClusters((current) =>
      current.map((cluster) =>
        cluster.id === id
          ? { ...cluster, favorite: !cluster.favorite }
          : cluster,
      ),
    );
  };

  const updateName = (value: string) => {
    setClusters((current) =>
      current.map((cluster) =>
        cluster.id === selectedCluster.id
          ? { ...cluster, name: value }
          : cluster,
      ),
    );
  };

  const createRun = () => {
    setRunning(true);
    window.setTimeout(() => setRunning(false), 1200);
  };

  return (
    <DemoShell
      active="clusters"
      eyebrow="CLUSTER LAB / 44.5"
      title="从相似中，看见结构。"
      description="比较不同 Embedding Type 的 Cluster Run、参数和质量，检查代表资产与完整成员。"
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
              value={selectedRun.type}
              onChange={(event) => {
                const match = RUNS.find((run) => run.type === event.target.value);
                if (match) setSelectedRunId(match.id);
              }}
            >
              <option value="native_multimodal">native_multimodal</option>
              <option value="mood_atmosphere">mood_atmosphere</option>
              <option value="visual_style">visual_style</option>
            </select>
          </label>
          <label>
            Cluster Run
            <select
              value={selectedRunId}
              onChange={(event) => setSelectedRunId(event.target.value)}
            >
              {RUNS.map((run) => (
                <option value={run.id} key={run.id}>
                  {run.id} · {run.created}
                </option>
              ))}
            </select>
          </label>
        </div>
        <div className="run-parameters">
          <span>
            <small>SAMPLES</small>
            <strong>{selectedRun.sampleCount}</strong>
          </span>
          <span>
            <small>CLUSTERS</small>
            <strong>{selectedRun.clusters}</strong>
          </span>
          <span>
            <small>NOISE RATIO</small>
            <strong>{(selectedRun.noise * 100).toFixed(1)}%</strong>
          </span>
          <span>
            <small>MIN CLUSTER</small>
            <strong>{selectedRun.minClusterSize}</strong>
          </span>
          <span>
            <small>MIN SAMPLES</small>
            <strong>{selectedRun.minSamples}</strong>
          </span>
          <span>
            <small>PCA</small>
            <strong>{selectedRun.pca}D</strong>
          </span>
        </div>
      </section>

      <div className="cluster-overview">
        <section className="cluster-map">
          <header>
            <div>
              <span className="eyebrow">PCA PROJECTION</span>
              <h2>二维分布观察</h2>
            </div>
            <small>仅用于观察，不参与聚类计算</small>
          </header>
          <div className="scatter-plot" aria-label="PCA 二维散点图">
            <span className="axis axis-x">PCA 1</span>
            <span className="axis axis-y">PCA 2</span>
            {DOTS.map(([left, top, group], index) => (
              <button
                className={`scatter-dot scatter-group-${group}`}
                style={{ left: `${left}%`, top: `${top}%` }}
                aria-label={`样本 ${index + 1}`}
                onClick={() =>
                  group >= 0 &&
                  setSelectedClusterId(
                    clusters[group % clusters.length].id,
                  )
                }
                key={`${left}-${top}`}
              />
            ))}
          </div>
          <footer>
            {clusters.slice(0, 4).map((cluster, index) => (
              <span key={cluster.id}>
                <i className={`scatter-group-${index}`} />
                {cluster.name}
              </span>
            ))}
            <span>
              <i className="scatter-group--1" />
              Noise
            </span>
          </footer>
        </section>

        <section className="cluster-cards">
          <header>
            <span className="eyebrow">CLUSTER CAPSULES</span>
            <h2>{clusters.length} 个语义分组</h2>
          </header>
          {clusters.map((cluster, index) => (
            <article
              className={selectedCluster.id === cluster.id ? "active" : ""}
              onClick={() => setSelectedClusterId(cluster.id)}
              key={cluster.id}
            >
              <div className={`cluster-number cluster-number-${index}`}>
                {String(index + 1).padStart(2, "0")}
              </div>
              <div>
                <small>{cluster.embeddingType}</small>
                <h3>{cluster.name}</h3>
                <p>{cluster.description}</p>
                <footer>
                  <span>{cluster.members} MEMBERS</span>
                  <span>{cluster.probability.toFixed(2)} AVG PROB.</span>
                </footer>
              </div>
              <button
                onClick={(event) => {
                  event.stopPropagation();
                  toggleFavorite(cluster.id);
                }}
                aria-label={cluster.favorite ? "取消收藏" : "收藏"}
              >
                {cluster.favorite ? "★" : "☆"}
              </button>
            </article>
          ))}
        </section>
      </div>

      <section className="cluster-detail">
        <header>
          <div>
            <span className="eyebrow">SELECTED CAPSULE</span>
            {renaming ? (
              <input
                value={selectedCluster.name}
                onChange={(event) => updateName(event.target.value)}
                onBlur={() => setRenaming(false)}
                autoFocus
              />
            ) : (
              <h2>{selectedCluster.name}</h2>
            )}
            <p>{selectedCluster.description}</p>
          </div>
          <div>
            <StatusBadge status="completed" />
            <button onClick={() => setRenaming(true)}>用户改名</button>
          </div>
        </header>
        <div className="representative-strip">
          {members.map((asset, index) => (
            <article key={asset.id}>
              <AssetThumb
                preview={asset.preview}
                name={asset.name}
                type={asset.type}
              />
              <span>REP {index + 1}</span>
              <strong>{asset.name}</strong>
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
          {members.map((asset, index) => (
            <div className="data-row member-row" key={asset.id}>
              <strong>{asset.name}</strong>
              <span>{asset.type}</span>
              <span className="membership-meter">
                <i style={{ width: `${94 - index * 7}%` }} />
                <b>{(0.94 - index * 0.07).toFixed(2)}</b>
              </span>
              <span>{asset.sourceFile}</span>
            </div>
          ))}
        </div>
      </section>
    </DemoShell>
  );
}
