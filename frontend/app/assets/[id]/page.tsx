"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useMemo, useState } from "react";
import DemoShell, {
  AssetThumb,
  StatusBadge,
} from "../../components/DemoShell";
import { DEMO_ASSETS } from "../../lib/demo-data";

export default function AssetDetailPage() {
  const pathname = usePathname();
  const assetId = pathname.split("/").filter(Boolean).at(-1);
  const asset =
    DEMO_ASSETS.find((item) => item.id === assetId) ?? DEMO_ASSETS[0];
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(asset.name);
  const [description, setDescription] = useState(asset.description);
  const [featureValues, setFeatureValues] = useState(
    Object.fromEntries(asset.features.map((feature) => [feature.key, feature.value])),
  );
  const [reprocessing, setReprocessing] = useState(false);
  const changed = useMemo(
    () =>
      name !== asset.name ||
      description !== asset.description ||
      asset.features.some(
        (feature) => featureValues[feature.key] !== feature.value,
      ),
    [asset.description, asset.features, asset.name, description, featureValues, name],
  );

  const reprocess = () => {
    setReprocessing(true);
    window.setTimeout(() => setReprocessing(false), 1100);
  };

  return (
    <DemoShell
      active="assets"
      eyebrow="ASSET DETAIL / 44.4"
      title={asset.name}
      description={`${asset.id} · ${asset.locator}`}
      actions={
        <>
          <Link className="secondary-action button-link" href="/assets">
            ← 返回列表
          </Link>
          <button
            className="secondary-action"
            disabled={reprocessing}
            onClick={reprocess}
          >
            {reprocessing ? "已加入队列…" : "重新处理"}
          </button>
          <button
            className="primary-action"
            onClick={() => setEditing((current) => !current)}
          >
            {editing ? "完成编辑" : "手动修改"}
          </button>
        </>
      }
    >
      <div className="asset-detail-hero">
        <div className="asset-detail-preview">
          <AssetThumb
            preview={asset.preview}
            name={asset.name}
            type={asset.type}
          />
          <div className="preview-meta">
            <StatusBadge status={asset.status} />
            <span>{asset.locator}</span>
          </div>
        </div>
        <section className="asset-core-info">
          <span className="eyebrow">SEMANTIC IDENTITY</span>
          <label>
            Asset Name
            {editing ? (
              <input value={name} onChange={(event) => setName(event.target.value)} />
            ) : (
              <strong>{name}</strong>
            )}
          </label>
          <label>
            Asset Description
            {editing ? (
              <textarea
                value={description}
                onChange={(event) => setDescription(event.target.value)}
                rows={5}
              />
            ) : (
              <p>{description}</p>
            )}
          </label>
          {changed && (
            <div className="unsaved-banner">
              <span>有未保存的人工修改</span>
              <button onClick={() => setEditing(false)}>保存并更新向量</button>
            </div>
          )}
        </section>
      </div>

      <div className="asset-detail-grid">
        <section className="source-inspector">
          <header>
            <span className="eyebrow">SOURCE</span>
            <h2>来源与原始位置</h2>
          </header>
          <dl>
            <div>
              <dt>Source File</dt>
              <dd>{asset.sourceFile}</dd>
            </div>
            <div>
              <dt>相对路径</dt>
              <dd>{asset.sourcePath}</dd>
            </div>
            <div>
              <dt>原始位置</dt>
              <dd>{asset.locator}</dd>
            </div>
            <div>
              <dt>所属 Cluster</dt>
              <dd>{asset.cluster ?? "尚未聚类"}</dd>
            </div>
          </dl>
          <blockquote>
            <span>关联段落</span>
            <p>{asset.sourceContext}</p>
          </blockquote>
        </section>

        <section className="embedding-inspector">
          <header>
            <span className="eyebrow">EMBEDDING GROUP</span>
            <h2>向量状态</h2>
          </header>
          {asset.embeddings.map((embedding) => (
            <div key={embedding.type}>
              <span>
                <i className={`embedding-dot ${embedding.status}`} />
                {embedding.type}
              </span>
              <StatusBadge status={embedding.status} />
              <small>REV {embedding.revision}</small>
            </div>
          ))}
        </section>
      </div>

      <section className="feature-editor">
        <header>
          <div>
            <span className="eyebrow">ASSET FEATURES</span>
            <h2>多维语义特征</h2>
          </div>
          <span>{asset.features.length} / 10 DIMENSIONS</span>
        </header>
        <div className="feature-table">
          <div className="feature-row feature-head">
            <span>维度</span>
            <span>Effective Value</span>
            <span>状态</span>
            <span>Confidence</span>
            <span>Evidence</span>
          </div>
          {asset.features.map((feature) => (
            <div className="feature-row" key={feature.key}>
              <strong>{feature.label}</strong>
              {editing ? (
                <textarea
                  value={featureValues[feature.key]}
                  onChange={(event) =>
                    setFeatureValues((current) => ({
                      ...current,
                      [feature.key]: event.target.value,
                    }))
                  }
                />
              ) : (
                <span>{featureValues[feature.key]}</span>
              )}
              <StatusBadge status={feature.status} />
              <span className="confidence-cell">
                <i style={{ width: `${feature.confidence * 100}%` }} />
                <b>{feature.confidence.toFixed(2)}</b>
              </span>
              <details>
                <summary>{feature.evidence.length} 条证据</summary>
                <ul>
                  {feature.evidence.map((evidence) => (
                    <li key={evidence}>{evidence}</li>
                  ))}
                </ul>
              </details>
            </div>
          ))}
        </div>
      </section>
    </DemoShell>
  );
}
