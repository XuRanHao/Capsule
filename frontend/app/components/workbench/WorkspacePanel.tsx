"use client";

import Link from "next/link";
import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import {
  createWorkspaceDirectory,
  createWorkspaceMarkdownFile,
  loadWorkspaceDirectories,
  type AssetRecord,
  type WorkspaceDirectoryRecord,
} from "../../lib/api";
import type { WorkspaceRecord } from "../../lib/workspaces";
import { WorkspaceSelect } from "../../lib/workspaces";

type Props = {
  workspaceId: string;
  workspaces: WorkspaceRecord[];
  loading: boolean;
  assets: AssetRecord[];
  creatingGraph: boolean;
  onWorkspaceChange: (workspaceId: string) => void;
  onCreateGraph: () => void;
  onAssetsRefresh: () => void;
};

type PendingFile = { relativePath: string; jobId: string };
type TreeFile = { name: string; relativePath: string; assetId?: string; pending?: boolean };
type TreeFolder = {
  name: string;
  path: string;
  folders: Map<string, TreeFolder>;
  files: TreeFile[];
};

function FolderIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className="workspace-folder-icon">
      <path d="M3.5 7.2h6l1.8 2h9.2v8.9a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2z" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
    </svg>
  );
}

function FileIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className="workspace-file-icon">
      <path d="M5 3.5h8l4 4V20.5H5z" fill="none" stroke="currentColor" strokeWidth="1.6" />
      <path d="M13 3.5v4h4M8 14h8M8 17h5" fill="none" stroke="currentColor" strokeWidth="1.6" />
    </svg>
  );
}

function PlusIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 5v14M5 12h14" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  );
}

function normalizePath(value: string) {
  return value.trim().replace(/\\/g, "/").replace(/^\/+|\/+$/g, "").replace(/\/+/g, "/");
}

function joinPath(...parts: string[]) {
  return normalizePath(parts.filter(Boolean).join("/"));
}

function treeFrom(
  assets: AssetRecord[],
  directories: WorkspaceDirectoryRecord[],
  pendingFiles: PendingFile[],
) {
  const root: TreeFolder = { name: "", path: "", folders: new Map(), files: [] };
  const ensureFolder = (path: string) => {
    let node = root;
    let currentPath = "";
    for (const name of normalizePath(path).split("/").filter(Boolean)) {
      currentPath = joinPath(currentPath, name);
      const child = node.folders.get(name) ?? {
        name,
        path: currentPath,
        folders: new Map<string, TreeFolder>(),
        files: [],
      };
      node.folders.set(name, child);
      node = child;
    }
    return node;
  };

  directories.forEach((directory) => ensureFolder(directory.path));
  const seenSources = new Set<string>();
  assets.forEach((asset) => {
    if (seenSources.has(asset.source_file_id)) return;
    seenSources.add(asset.source_file_id);
    const relativePath = normalizePath(asset.source_file.relative_path || asset.file_name);
    const parts = relativePath.split("/");
    const fileName = parts.pop() || asset.file_name;
    ensureFolder(parts.join("/")).files.push({ name: fileName, relativePath, assetId: asset.asset_id });
  });
  pendingFiles.forEach((file) => {
    const parts = file.relativePath.split("/");
    const fileName = parts.pop() || file.relativePath;
    ensureFolder(parts.join("/")).files.push({ name: fileName, relativePath: file.relativePath, pending: true });
  });
  return root;
}

function TreeChildren({ folder }: { folder: TreeFolder }) {
  const folders = [...folder.folders.values()].sort((left, right) => left.name.localeCompare(right.name));
  const files = [...folder.files].sort((left, right) => left.name.localeCompare(right.name));
  return (
    <ul className="workspace-tree-list">
      {folders.map((child) => (
        <li key={child.path}>
          <details open>
            <summary>
              <FolderIcon />
              <span>{child.name}</span>
              <small>{child.files.length + child.folders.size}</small>
            </summary>
            <TreeChildren folder={child} />
          </details>
        </li>
      ))}
      {files.map((file) => (
        <li key={file.relativePath}>
          {file.assetId ? (
            <Link href={`/assets/${encodeURIComponent(file.assetId)}`} className="workspace-tree-file">
              <FileIcon /><span>{file.name}</span>
            </Link>
          ) : (
            <div className="workspace-tree-file workspace-tree-file-pending">
              <FileIcon /><span>{file.name}</span><small>处理中</small>
            </div>
          )}
        </li>
      ))}
    </ul>
  );
}

export default function WorkspacePanel({
  workspaceId,
  workspaces,
  loading,
  assets,
  creatingGraph,
  onWorkspaceChange,
  onCreateGraph,
  onAssetsRefresh,
}: Props) {
  const [directories, setDirectories] = useState<WorkspaceDirectoryRecord[]>([]);
  const [pendingFiles, setPendingFiles] = useState<PendingFile[]>([]);
  const [dialog, setDialog] = useState<"file" | "folder" | null>(null);
  const [folderPath, setFolderPath] = useState("");
  const [fileName, setFileName] = useState("untitled.md");
  const [fileContent, setFileContent] = useState("");
  const [saving, setSaving] = useState(false);
  const [treeError, setTreeError] = useState<string | null>(null);
  const currentWorkspace = workspaces.find((item) => item.workspace_id === workspaceId);

  const refreshDirectories = useCallback(async () => {
    if (!workspaceId) return;
    try {
      const response = await loadWorkspaceDirectories(workspaceId);
      setDirectories(response.items);
      setTreeError(null);
    } catch (error) {
      setDirectories([]);
      setTreeError(error instanceof Error ? error.message : "目录加载失败");
    }
  }, [workspaceId]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void refreshDirectories();
      setPendingFiles([]);
    }, 0);
    return () => window.clearTimeout(timer);
  }, [refreshDirectories]);

  const visiblePendingFiles = useMemo(
    () => pendingFiles.filter((pending) => !assets.some(
      (asset) => normalizePath(asset.source_file.relative_path) === pending.relativePath,
    )),
    [assets, pendingFiles],
  );

  useEffect(() => {
    if (!visiblePendingFiles.length) return;
    const timer = window.setInterval(onAssetsRefresh, 4000);
    return () => window.clearInterval(timer);
  }, [onAssetsRefresh, visiblePendingFiles.length]);

  const tree = useMemo(
    () => treeFrom(assets, directories, visiblePendingFiles),
    [assets, directories, visiblePendingFiles],
  );

  const closeDialog = (force = false) => {
    if (saving && !force) return;
    setDialog(null);
    setFolderPath("");
    setFileName("untitled.md");
    setFileContent("");
  };

  const saveFolder = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const relativePath = normalizePath(folderPath);
    if (!relativePath) return;
    setSaving(true);
    try {
      await createWorkspaceDirectory({ workspace_id: workspaceId, path: relativePath });
      await refreshDirectories();
      closeDialog(true);
    } catch (error) {
      setTreeError(error instanceof Error ? error.message : "新建文件夹失败");
    } finally {
      setSaving(false);
    }
  };

  const saveFile = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const rawName = fileName.trim();
    const name = rawName.endsWith(".md") ? rawName : `${rawName}.md`;
    const relativePath = joinPath(folderPath, name);
    if (!relativePath || name === ".md") return;
    setSaving(true);
    try {
      const parentPath = relativePath.split("/").slice(0, -1).join("/");
      if (parentPath) await createWorkspaceDirectory({ workspace_id: workspaceId, path: parentPath });
      const created = await createWorkspaceMarkdownFile({ workspace_id: workspaceId, relative_path: relativePath, content: fileContent });
      setPendingFiles((current) => [...current, { relativePath, jobId: created.job_id }]);
      await refreshDirectories();
      closeDialog(true);
      window.setTimeout(onAssetsRefresh, 1200);
    } catch (error) {
      setTreeError(error instanceof Error ? error.message : "文件保存失败");
    } finally {
      setSaving(false);
    }
  };

  return (
    <aside className="workbench-panel workspace-panel" aria-label="工作空间">
      <div className="panel-heading">
        <div><span className="panel-kicker">WORKSPACE</span><h1>{currentWorkspace?.name || "我的工作空间"}</h1></div>
        <WorkspaceSelect workspaceId={workspaceId} workspaces={workspaces} loading={loading} onChange={onWorkspaceChange} />
      </div>

      <div className="workspace-summary"><span>{assets.length}</span><p>个素材已进入当前知识空间</p></div>

      <div className="workspace-create-actions" aria-label="创建资源">
        <button type="button" className="workspace-create-primary" onClick={() => setDialog("file")}><PlusIcon /> 新建文件</button>
        <button type="button" className="workspace-create-icon" aria-label="新建文件夹" title="新建文件夹" onClick={() => setDialog("folder")}><FolderIcon /></button>
        <button type="button" className="workspace-create-graph" onClick={onCreateGraph} disabled={creatingGraph}>{creatingGraph ? "正在创建…" : "新建图谱"}</button>
      </div>

      <div className="workspace-actions"><Link href="/import" className="workspace-import-link">导入文件或文件夹</Link><Link href="/assets" className="workspace-library-link">打开素材库</Link></div>

      <section className="workspace-tree" aria-label="工作空间文件树">
        <div className="tree-section-heading"><span>文件资源</span><small>{assets.length + visiblePendingFiles.length} ITEMS</small></div>
        {treeError && <p className="workspace-tree-error" role="status">{treeError}</p>}
        {tree.folders.size || tree.files.length ? <TreeChildren folder={tree} /> : (
          <div className="workspace-empty"><span>尚无文件或文件夹</span><p>新建 Markdown 文件后会自动进入素材处理链路。</p></div>
        )}
      </section>

      {dialog === "folder" && (
        <div className="workspace-dialog-backdrop" role="presentation" onMouseDown={closeDialog}>
          <form className="workspace-dialog" aria-label="新建文件夹" onSubmit={saveFolder} onMouseDown={(event) => event.stopPropagation()}>
            <header><span className="panel-kicker">NEW FOLDER</span><h2>新建文件夹</h2></header>
            <label>文件夹路径<input autoFocus value={folderPath} onChange={(event) => setFolderPath(event.target.value)} placeholder="例如：研究/访谈" required disabled={saving} /></label>
            <p>使用 “/” 建立多级目录，父目录会自动创建。</p>
            <footer><button type="button" onClick={closeDialog} disabled={saving}>取消</button><button type="submit" disabled={saving}>{saving ? "正在创建…" : "创建文件夹"}</button></footer>
          </form>
        </div>
      )}

      {dialog === "file" && (
        <div className="workspace-dialog-backdrop" role="presentation" onMouseDown={closeDialog}>
          <form className="workspace-dialog workspace-file-dialog" aria-label="新建 Markdown 文件" onSubmit={saveFile} onMouseDown={(event) => event.stopPropagation()}>
            <header><span className="panel-kicker">NEW FILE</span><h2>新建 Markdown 文件</h2></header>
            <label>所在文件夹（可选）<input value={folderPath} onChange={(event) => setFolderPath(event.target.value)} placeholder="例如：研究/访谈" disabled={saving} /></label>
            <label>文件名<input autoFocus value={fileName} onChange={(event) => setFileName(event.target.value)} placeholder="untitled.md" required disabled={saving} /></label>
            <label>内容<textarea value={fileContent} onChange={(event) => setFileContent(event.target.value)} placeholder="从这里开始记录…" rows={9} disabled={saving} /></label>
            <p>保存后将作为素材进入内容理解、向量化与入库流程。</p>
            <footer><button type="button" onClick={closeDialog} disabled={saving}>取消</button><button type="submit" disabled={saving}>{saving ? "正在保存…" : "保存并入库"}</button></footer>
          </form>
        </div>
      )}
    </aside>
  );
}
