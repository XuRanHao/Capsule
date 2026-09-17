"use client";

import { useMemo, useState } from "react";
import type { AssetRecord } from "../../lib/api";

type Props = { assets: AssetRecord[]; selectedId: string | null; onSelect: (assetId: string) => void };

type GraphNode = { id: string; label: string; kind: string; sourceFileId: string; x: number; y: number };

const POSITIONS = [
  [50, 45], [18, 27], [83, 22], [24, 73], [77, 76], [49, 88], [9, 57], [91, 56],
];

function materialLabel(asset: AssetRecord) {
  return (asset.asset_name || asset.file_name || "未命名素材").slice(0, 16);
}

export default function GraphCanvas({ assets, selectedId, onSelect }: Props) {
  const [scale, setScale] = useState(1);
  const nodes = useMemo<GraphNode[]>(() => assets.slice(0, 8).map((asset, index) => ({
    id: asset.asset_id,
    label: materialLabel(asset),
    kind: asset.asset_type, sourceFileId: asset.source_file_id,
    x: POSITIONS[index][0],
    y: POSITIONS[index][1],
  })), [assets]);
  const links = useMemo(() => nodes.flatMap((node, index) =>
    nodes.slice(index + 1)
      .filter((candidate) => candidate.sourceFileId === node.sourceFileId)
      .map((candidate) => ({ from: node.id, to: candidate.id })),
  ), [nodes]);

  return (
    <section className="workbench-graph" aria-label="素材关系图谱">
      <header className="graph-header">
        <div>
          <span className="panel-kicker">KNOWLEDGE GRAPH</span>
          <h2>素材关系图谱</h2>
        </div>
        <div className="graph-toolbar" aria-label="图谱缩放控制">
          <button type="button" aria-label="缩小图谱" onClick={() => setScale((value) => Math.max(0.7, value - 0.1))}>−</button>
          <output>{Math.round(scale * 100)}%</output>
          <button type="button" aria-label="放大图谱" onClick={() => setScale((value) => Math.min(1.4, value + 0.1))}>＋</button>
          <button type="button" className="graph-reset" onClick={() => setScale(1)}>适配</button>
        </div>
      </header>

      <div className="graph-legend" aria-label="图谱图例">
        <span><i className="node-dot image" />视觉素材</span>
        <span><i className="node-dot text" />文本与片段</span>
        <span><i className="node-line" />素材关联</span>
      </div>

      {nodes.length ? (
        <div className="graph-stage">
          <div className="graph-world" style={{ transform: `scale(${scale})` }}>
            <svg className="graph-links" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
              {links.map((link) => {
                const from = nodes.find((node) => node.id === link.from)!;
                const to = nodes.find((node) => node.id === link.to)!;
                return <line key={`${link.from}-${link.to}`} x1={from.x} y1={from.y} x2={to.x} y2={to.y} />;
              })}
            </svg>
            {nodes.map((node) => (
              <button
                key={node.id}
                type="button"
                className={`graph-node ${node.kind === "image" || node.kind === "video_segment" ? "visual" : "textual"} ${selectedId === node.id ? "selected" : ""}`}
                style={{ left: `${node.x}%`, top: `${node.y}%` }}
                onClick={() => onSelect(node.id)}
                title={`查看 ${node.label}`}
              >
                <span>{node.kind === "image" ? "IMG" : node.kind === "video_segment" ? "VID" : "TXT"}</span>
                <strong>{node.label}</strong>
              </button>
            ))}
          </div>
        </div>
      ) : (
        <div className="graph-empty">
          <div className="graph-empty-orbit" aria-hidden="true" />
          <h3>等待第一批素材</h3>
          <p>素材入库后，会按来源与上下文组织成可浏览的关系图谱。</p>
        </div>
      )}
      <footer className="graph-footer">显示 {nodes.length} 个节点 · {links.length} 条同源文件关系</footer>
    </section>
  );
}
