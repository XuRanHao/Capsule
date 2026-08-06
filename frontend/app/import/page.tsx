"use client";

import { DragEvent, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import DemoShell from "../components/DemoShell";

type QueuedFile = {
  file: File;
  name: string;
  path: string;
  type: "markdown" | "image" | "video";
  size: number;
};

type ImportPhase = "creating" | "uploading" | "starting";

type UploadProgress = {
  completed: number;
  loadedBytes: number;
  totalBytes: number;
  currentName: string;
};

const folderInputProps = {
  webkitdirectory: "",
  directory: "",
} as React.InputHTMLAttributes<HTMLInputElement>;

type DirectoryPickerWindow = Window & {
  showDirectoryPicker?: () => Promise<FileSystemDirectoryHandle>;
};

type DirectoryHandleWithValues = FileSystemDirectoryHandle & {
  values: () => AsyncIterable<FileSystemHandle>;
};

function asQueuedFile(file: File, path: string): QueuedFile | null {
  const extension = file.name.split(".").pop()?.toLowerCase() ?? "";
  const type =
    extension === "md"
      ? "markdown"
      : ["mp4", "mov"].includes(extension)
        ? "video"
        : ["jpg", "jpeg", "png", "webp"].includes(extension)
          ? "image"
          : null;
  if (!type) return null;
  return { file, name: file.name, path, type, size: file.size };
}

function normalizeFiles(files: FileList | File[]): QueuedFile[] {
  return Array.from(files)
    .map((file) => asQueuedFile(file, file.webkitRelativePath || file.name))
    .filter((item): item is QueuedFile => item !== null);
}

async function collectDirectoryFiles(
  directory: FileSystemDirectoryHandle,
  parentPath = directory.name,
): Promise<QueuedFile[]> {
  const files: QueuedFile[] = [];
  const entries = (directory as DirectoryHandleWithValues).values();
  for await (const entry of entries) {
    const relativePath = `${parentPath}/${entry.name}`;
    if (entry.kind === "file") {
      const queued = asQueuedFile(
        await (entry as FileSystemFileHandle).getFile(),
        relativePath,
      );
      if (queued) files.push(queued);
    } else {
      files.push(
        ...(await collectDirectoryFiles(
          entry as FileSystemDirectoryHandle,
          relativePath,
        )),
      );
    }
  }
  return files;
}

function formatBytes(bytes: number) {
  if (bytes > 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

function endpoint(path: string) {
  const base = process.env.NEXT_PUBLIC_CAPSULE_API_BASE_URL ?? "";
  return `${base.replace(/\/$/, "")}${path}`;
}

function uploadFile(
  {
    jobId,
    workspaceId,
    item,
    onProgress,
  }: {
    jobId: string;
    workspaceId: string;
    item: QueuedFile;
    onProgress: (loadedBytes: number) => void;
  },
): Promise<void> {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.set("workspace_id", workspaceId);
    form.set("relative_path", item.path);
    form.set("file", item.file, item.name);

    const request = new XMLHttpRequest();
    request.open(
      "POST",
      endpoint(`/api/v1/import-jobs/${encodeURIComponent(jobId)}/files`),
    );
    request.upload.onprogress = (event) => {
      onProgress(Math.min(event.loaded, item.size));
    };
    request.onerror = () => reject(new Error(`上传 ${item.name} 时网络中断`));
    request.onabort = () => reject(new Error(`上传 ${item.name} 已取消`));
    request.onload = () => {
      if (request.status >= 200 && request.status < 300) {
        onProgress(item.size);
        resolve();
        return;
      }
      let message = `上传 ${item.name} 失败`;
      try {
        const body = JSON.parse(request.responseText) as {
          detail?: { message?: string };
        };
        message = body.detail?.message || message;
      } catch {
        // Keep the file-specific fallback when the proxy returns a non-JSON error.
      }
      reject(new Error(message));
    };
    request.send(form);
  });
}

export default function ImportPage() {
  const router = useRouter();
  const fileInput = useRef<HTMLInputElement>(null);
  const folderInput = useRef<HTMLInputElement>(null);
  const [files, setFiles] = useState<QueuedFile[]>([]);
  const [dragging, setDragging] = useState(false);
  const [starting, setStarting] = useState(false);
  const [importError, setImportError] = useState<string | null>(null);
  const [importPhase, setImportPhase] = useState<ImportPhase | null>(null);
  const [uploadProgress, setUploadProgress] = useState<UploadProgress | null>(null);

  const counts = useMemo(
    () => ({
      markdown: files.filter((item) => item.type === "markdown").length,
      image: files.filter((item) => item.type === "image").length,
      video: files.filter((item) => item.type === "video").length,
      bytes: files.reduce((total, item) => total + item.size, 0),
    }),
    [files],
  );

  const addFiles = (incoming: FileList | File[]) => {
    addQueuedFiles(normalizeFiles(incoming));
  };

  const addQueuedFiles = (incoming: QueuedFile[]) => {
    setFiles((current) => {
      const seen = new Set(current.map((item) => item.path));
      return [...current, ...incoming.filter((item) => !seen.has(item.path))];
    });
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    addFiles(event.dataTransfer.files);
  };

  const selectDirectory = async () => {
    setImportError(null);
    const picker = (window as DirectoryPickerWindow).showDirectoryPicker;
    if (!picker) {
      folderInput.current?.click();
      return;
    }
    try {
      addQueuedFiles(await collectDirectoryFiles(await picker()));
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return;
      setImportError(error instanceof Error ? error.message : "读取文件夹失败");
    }
  };

  const startImport = async () => {
    if (!files.length) return;
    setStarting(true);
    setImportError(null);
    setImportPhase("creating");
    setUploadProgress(null);
    try {
      const workspaceId = "workspace_demo";
      const created = await fetch(endpoint("/api/v1/import-jobs"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ workspace_id: workspaceId }),
      });
      if (!created.ok) {
        const body = await created.json().catch(() => null);
        throw new Error(body?.detail?.message || "创建导入任务失败");
      }
      const { job_id: jobId } = (await created.json()) as { job_id: string };

      setImportPhase("uploading");
      const loadedByPath = new Map<string, number>();
      let completed = 0;
      let nextIndex = 0;
      const publishProgress = (item: QueuedFile, loadedBytes: number) => {
        loadedByPath.set(item.path, loadedBytes);
        setUploadProgress({
          completed,
          loadedBytes: Array.from(loadedByPath.values()).reduce(
            (total, loaded) => total + loaded,
            0,
          ),
          totalBytes: counts.bytes,
          currentName: item.name,
        });
      };
      const uploadItem = async (item: QueuedFile) => {
        let lastError: Error | null = null;
        for (let attempt = 0; attempt < 3; attempt += 1) {
          try {
            await uploadFile({
              jobId,
              workspaceId,
              item,
              onProgress: (loadedBytes) => publishProgress(item, loadedBytes),
            });
            lastError = null;
            break;
          } catch (error) {
            lastError = error instanceof Error ? error : new Error(`上传 ${item.name} 失败`);
            publishProgress(item, 0);
            if (attempt < 2) {
              await new Promise((resolve) => window.setTimeout(resolve, 600 * (attempt + 1)));
            }
          }
        }
        if (lastError) throw lastError;
        completed += 1;
        publishProgress(item, item.size);
      };
      const uploadWorker = async () => {
        while (nextIndex < files.length) {
          const item = files[nextIndex];
          nextIndex += 1;
          await uploadItem(item);
        }
      };
      await Promise.all(
        Array.from({ length: Math.min(4, files.length) }, () => uploadWorker()),
      );

      setImportPhase("starting");
      const started = await fetch(
        endpoint(`/api/v1/import-jobs/${encodeURIComponent(jobId)}/complete`),
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ workspace_id: workspaceId }),
        },
      );
      if (!started.ok) {
        const body = await started.json().catch(() => null);
        throw new Error(body?.detail?.message || "启动资产化任务失败");
      }
      router.push(`/tasks?job_id=${encodeURIComponent(jobId)}`);
    } catch (error) {
      setImportError(error instanceof Error ? error.message : "创建导入任务失败");
    } finally {
      setStarting(false);
      setImportPhase(null);
      setUploadProgress(null);
    }
  };

  return (
    <DemoShell
      active="import"
      eyebrow="IMPORT / 44.1"
      title="把散落的素材，带回同一个地方。"
      description="保留文件夹层级和相对路径，自动识别 Markdown、图片与视频，提交后进入异步处理链路。"
      actions={
        <button className="secondary-action" onClick={() => setFiles([])}>
          清空队列
        </button>
      }
    >
      <div className="import-overview">
        <div>
          <small>待导入文件</small>
          <strong>{files.length}</strong>
          <span>{formatBytes(counts.bytes)}</span>
        </div>
        <div>
          <small>Markdown</small>
          <strong>{counts.markdown}</strong>
          <span>语义 Block</span>
        </div>
        <div>
          <small>图片</small>
          <strong>{counts.image}</strong>
          <span>原图 + 上下文</span>
        </div>
        <div>
          <small>视频</small>
          <strong>{counts.video}</strong>
          <span>场景切分</span>
        </div>
      </div>

      <div className="import-workspace">
        <div
          className={`drop-zone ${dragging ? "dragging" : ""}`}
          onDragEnter={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragOver={(event) => event.preventDefault()}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
        >
          <div className="drop-orbit" aria-hidden="true">
            <span>＋</span>
          </div>
          <h2>拖入文件或整个文件夹</h2>
          <p>目录结构会写入 Source File，重复 SHA-256 文件会自动跳过。</p>
          <div>
            <button onClick={() => fileInput.current?.click()}>
              选择文件
            </button>
            <button onClick={selectDirectory}>
              选择文件夹
            </button>
          </div>
          <input
            ref={fileInput}
            type="file"
            multiple
            accept=".md,.jpg,.jpeg,.png,.webp,.mp4,.mov"
            onChange={(event) => event.target.files && addFiles(event.target.files)}
            hidden
          />
          <input
            ref={folderInput}
            type="file"
            multiple
            {...folderInputProps}
            onChange={(event) => event.target.files && addFiles(event.target.files)}
            hidden
          />
        </div>

        <aside className="format-panel">
          <span className="eyebrow">SUPPORTED</span>
          <h3>支持格式</h3>
          <div>
            <span>MD</span>
            <p>
              <strong>Markdown</strong>
              <small>.md</small>
            </p>
          </div>
          <div>
            <span>IMG</span>
            <p>
              <strong>图片</strong>
              <small>.jpg · .png · .webp</small>
            </p>
          </div>
          <div>
            <span>VID</span>
            <p>
              <strong>视频</strong>
              <small>.mp4 · .mov</small>
            </p>
          </div>
          <footer>
            单文件建议不超过 2 GB
            <br />
            UTF-8 Markdown 获得最佳效果
          </footer>
        </aside>
      </div>

      <section className="file-queue">
        <header>
          <div>
            <span className="eyebrow">QUEUE / RELATIVE PATH</span>
            <h2>导入清单</h2>
          </div>
        </header>
        <div className="data-table">
          <div className="data-row data-head">
            <span>类型</span>
            <span>文件</span>
            <span>相对路径</span>
            <span>大小</span>
            <span />
          </div>
          {files.map((file) => (
            <div className="data-row" key={file.path}>
              <span className={`file-type-mark file-${file.type}`}>
                {file.type.slice(0, 3).toUpperCase()}
              </span>
              <strong>{file.name}</strong>
              <span className="path-cell">{file.path}</span>
              <span>{formatBytes(file.size)}</span>
              <button
                aria-label={`移除 ${file.name}`}
                onClick={() =>
                  setFiles((current) =>
                    current.filter((item) => item.path !== file.path),
                  )
                }
              >
                ×
              </button>
            </div>
          ))}
          {!files.length && <div className="table-empty">尚未选择文件</div>}
        </div>
      </section>

      <div className="import-submit-bar">
        <div>
          <small>WORKSPACE</small>
          <strong>workspace_demo</strong>
          <span>
            {importPhase === "creating"
              ? "正在创建导入任务"
              : importPhase === "starting"
                ? "上传完成，正在启动处理任务"
                : importPhase === "uploading" && uploadProgress
                  ? `正在上传 ${uploadProgress.completed}/${files.length} · ${Math.round((uploadProgress.loadedBytes / Math.max(1, uploadProgress.totalBytes)) * 100)}% · ${uploadProgress.currentName}`
                  : `${files.length} 个 Source File 将进入处理队列`}
          </span>
          {importError && <span className="import-error">{importError}</span>}
        </div>
        <button disabled={!files.length || starting} onClick={startImport}>
          {importPhase === "creating"
            ? "正在创建任务…"
            : importPhase === "uploading"
              ? "正在上传…"
              : importPhase === "starting"
                ? "正在启动处理…"
                : "开始处理"}
          <b>↗</b>
        </button>
      </div>
    </DemoShell>
  );
}
