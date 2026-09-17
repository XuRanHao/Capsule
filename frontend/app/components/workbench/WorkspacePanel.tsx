"use client";

import Link from "next/link";
import type { AssetRecord } from "../../lib/api";
import type { WorkspaceRecord } from "../../lib/workspaces";
import { WorkspaceSelect } from "../../lib/workspaces";

type Props = {
  workspaceId: string;
  workspaces: WorkspaceRecord[];
  loading: boolean;
  assets: AssetRecord[];
  onWorkspaceChange: (workspaceId: string) => void;
};

function FileIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className="workspace-file-icon">
      <path d="M5 3.5h8l4 4V20.5H5z" fill="none" stroke="currentColor" strokeWidth="1.6" />
      <path d="M13 3.5v4h4" fill="none" stroke="currentColor" strokeWidth="1.6" />
      <path d="M8 14h8M8 17h5" fill="none" stroke="currentColor" strokeWidth="1.6" />
    </svg>
  );
}

export default function WorkspacePanel({
  workspaceId,
  workspaces,
  loading,
  assets,
  onWorkspaceChange,
}: Props) {
  const recentAssets = assets.slice(0, 7);
  const currentWorkspace = workspaces.find((item) => item.workspace_id === workspaceId);

  return (
    <aside className="workbench-panel workspace-panel" aria-label="工作空间">
      <div className="panel-heading">
        <div>
          <span className="panel-kicker">WORKSPACE</span>
          <h1>{currentWorkspace?.name || "我的工作空间"}</h1>
        </div>
        <WorkspaceSelect
          workspaceId={workspaceId}
          workspaces={workspaces}
          loading={loading}
          onChange={onWorkspaceChange}
        />
      </div>

      <div className="workspace-summary">
        <span>{assets.length}</span>
        <p>个素材已进入当前知识空间</p>
      </div>

      <div className="workspace-actions">
        <Link href="/import" className="workspace-import-link">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 5v10m-5-5h10M5 20h14" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" /></svg>
          导入素材
        </Link>
        <Link href="/assets" className="workspace-library-link">打开素材库</Link>
      </div>

      <section className="workspace-tree" aria-label="最近入库素材">
        <div className="tree-section-heading">
          <span>最近素材</span>
          <small>{assets.length} ITEMS</small>
        </div>
        {recentAssets.length ? (
          <ul>
            {recentAssets.map((asset) => (
              <li key={asset.asset_id}>
                <Link href={`/assets/${encodeURIComponent(asset.asset_id)}`}>
                  <FileIcon />
                  <span>
                    <strong>{asset.asset_name || asset.file_name}</strong>
                    <small>{asset.asset_type.replace("_", " · ")}</small>
                  </span>
                </Link>
              </li>
            ))}
          </ul>
        ) : (
          <div className="workspace-empty">
            <span>尚无素材</span>
            <p>导入文件后，这里会生成可追溯的知识关系。</p>
          </div>
        )}
      </section>
    </aside>
  );
}
